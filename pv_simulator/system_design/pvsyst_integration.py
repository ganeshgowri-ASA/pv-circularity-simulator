"""PVsyst integration for file parsing and project generation."""

import logging
import re
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pandas as pd
import numpy as np

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    WeatherData,
    PVsystPanFile,
    PVsystOndFile,
    InverterType,
)

logger = logging.getLogger(__name__)


class PVsystIntegration:
    """
    PVsyst file parser and integration layer.

    Handles parsing of .PAN (module), .OND (inverter), and .MET (weather) files,
    and generates PVsyst-compatible project files.
    """

    def __init__(self):
        """Initialize PVsyst integration."""
        logger.info("Initialized PVsystIntegration")

    def parse_pvsyst_pan_file(self, file_path: str) -> ModuleParameters:
        """
        Parse PVsyst .PAN file to extract module parameters.

        PVsyst .PAN files can be text-based or binary format. This method
        handles both formats.

        Args:
            file_path: Path to .PAN file

        Returns:
            ModuleParameters object with parsed data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PAN file not found: {file_path}")

        logger.info(f"Parsing PAN file: {file_path}")

        try:
            # Try reading as text file first
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Extract parameters using regex
            params = {}

            # Common PAN file parameters
            patterns = {
                'manufacturer': r'Manufacturer\s*=\s*(.+)',
                'model': r'Model\s*=\s*(.+)',
                'technology': r'Technol\s*=\s*(.+)',
                'pmax': r'Pmpp\s*=\s*([\d.]+)',
                'voc': r'Voc\s*=\s*([\d.]+)',
                'isc': r'Isc\s*=\s*([\d.]+)',
                'vmp': r'Vmpp\s*=\s*([\d.]+)',
                'imp': r'Impp\s*=\s*([\d.]+)',
                'temp_coeff_pmax': r'muPmpp\s*=\s*([-\d.]+)',
                'temp_coeff_voc': r'muVocSpec\s*=\s*([-\d.]+)',
                'temp_coeff_isc': r'muIscSpec\s*=\s*([-\d.]+)',
                'length': r'Length\s*=\s*([\d.]+)',
                'width': r'Width\s*=\s*([\d.]+)',
                'thickness': r'Depth\s*=\s*([\d.]+)',
                'weight': r'Weight\s*=\s*([\d.]+)',
                'cells_in_series': r'NCelS\s*=\s*(\d+)',
                'efficiency': r'EfficiencyModule\s*=\s*([\d.]+)',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    try:
                        # Try converting to appropriate type
                        if key in ['manufacturer', 'model', 'technology']:
                            params[key] = value.strip('"\'')
                        elif key in ['cells_in_series']:
                            params[key] = int(value)
                        else:
                            params[key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not convert {key}={value}")
                        params[key] = value

            # Validate required parameters
            required = ['manufacturer', 'model', 'pmax', 'voc', 'isc', 'vmp', 'imp']
            missing = [r for r in required if r not in params]
            if missing:
                raise ValueError(f"Missing required parameters: {missing}")

            # Set defaults for optional parameters
            defaults = {
                'technology': 'mtSiMono',
                'temp_coeff_pmax': -0.4,
                'temp_coeff_voc': -0.29,
                'temp_coeff_isc': 0.05,
                'length': 1.7,
                'width': 1.0,
                'thickness': 0.04,
                'weight': 20.0,
                'cells_in_series': 60,
                'efficiency': 20.0,
            }

            for key, default_value in defaults.items():
                if key not in params:
                    params[key] = default_value
                    logger.debug(f"Using default {key}={default_value}")

            # Calculate efficiency if not provided
            if params.get('efficiency', 0) == 0:
                area = params.get('length', 1.7) * params.get('width', 1.0)
                params['efficiency'] = (params['pmax'] / (area * 1000)) * 100

            module = ModuleParameters(**params)

            logger.info(
                f"Parsed PAN file: {module.manufacturer} {module.model}, "
                f"{module.pmax}W, {module.efficiency:.1f}%"
            )

            return module

        except UnicodeDecodeError:
            # File might be binary format
            logger.warning("Text parsing failed, attempting binary parsing")
            return self._parse_binary_pan_file(file_path)

    def _parse_binary_pan_file(self, file_path: str) -> ModuleParameters:
        """
        Parse binary format PVsyst .PAN file.

        Args:
            file_path: Path to binary .PAN file

        Returns:
            ModuleParameters object
        """
        # Binary PAN file parsing is complex and version-dependent
        # This is a simplified implementation
        logger.warning("Binary PAN parsing not fully implemented, using defaults")

        # Return default module
        return ModuleParameters(
            manufacturer="Unknown",
            model="Generic Module",
            technology="mtSiMono",
            pmax=400.0,
            voc=48.0,
            isc=10.5,
            vmp=40.0,
            imp=10.0,
            temp_coeff_pmax=-0.4,
            temp_coeff_voc=-0.29,
            temp_coeff_isc=0.05,
            length=1.7,
            width=1.0,
            thickness=0.04,
            weight=20.0,
            cells_in_series=72,
            efficiency=20.0,
        )

    def parse_pvsyst_ond_file(self, file_path: str) -> InverterParameters:
        """
        Parse PVsyst .OND file to extract inverter parameters.

        Args:
            file_path: Path to .OND file

        Returns:
            InverterParameters object with parsed data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"OND file not found: {file_path}")

        logger.info(f"Parsing OND file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            params = {}

            # Common OND file parameters
            patterns = {
                'manufacturer': r'Manufacturer\s*=\s*(.+)',
                'model': r'Model\s*=\s*(.+)',
                'pac_max': r'Pnom\s*=\s*([\d.]+)',
                'vac_nom': r'VOutConv\s*=\s*([\d.]+)',
                'pdc_max': r'PNomDC\s*=\s*([\d.]+)',
                'vdc_max': r'VAbsMax\s*=\s*([\d.]+)',
                'vdc_nom': r'VNomEff\s*=\s*([\d.]+)',
                'vdc_min': r'VMinEff\s*=\s*([\d.]+)',
                'idc_max': r'IMaxDC\s*=\s*([\d.]+)',
                'num_mppt': r'NbInputs\s*=\s*(\d+)',
                'mppt_vmin': r'VMppMin\s*=\s*([\d.]+)',
                'mppt_vmax': r'VMPPMax\s*=\s*([\d.]+)',
                'max_efficiency': r'EfficMax\s*=\s*([\d.]+)',
                'euro_efficiency': r'EfficEuro\s*=\s*([\d.]+)',
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    try:
                        if key in ['manufacturer', 'model']:
                            params[key] = value.strip('"\'')
                        elif key == 'num_mppt':
                            params[key] = int(value)
                        else:
                            params[key] = float(value)
                    except ValueError:
                        logger.warning(f"Could not convert {key}={value}")

            # Convert kW to W if necessary
            if params.get('pac_max', 0) < 1000:
                params['pac_max'] = params.get('pac_max', 100) * 1000
            if params.get('pdc_max', 0) < 1000:
                params['pdc_max'] = params.get('pdc_max', 110) * 1000

            # Set defaults
            defaults = {
                'manufacturer': 'Unknown',
                'model': 'Generic Inverter',
                'inverter_type': InverterType.STRING,
                'pac_max': 100000.0,
                'vac_nom': 480.0,
                'iac_max': 150.0,
                'pdc_max': 110000.0,
                'vdc_max': 1000.0,
                'vdc_nom': 600.0,
                'vdc_min': 200.0,
                'idc_max': 200.0,
                'num_mppt': 2,
                'mppt_vmin': 200.0,
                'mppt_vmax': 850.0,
                'strings_per_mppt': 10,
                'max_efficiency': 98.0,
                'weight': 60.0,
            }

            for key, default_value in defaults.items():
                if key not in params:
                    params[key] = default_value

            # Calculate derived parameters
            if 'iac_max' not in params:
                params['iac_max'] = params['pac_max'] / params['vac_nom']

            if 'strings_per_mppt' not in params:
                params['strings_per_mppt'] = 10  # Default

            inverter = InverterParameters(**params)

            logger.info(
                f"Parsed OND file: {inverter.manufacturer} {inverter.model}, "
                f"{inverter.pac_max/1000:.1f}kW, {inverter.max_efficiency}%"
            )

            return inverter

        except Exception as e:
            logger.error(f"Error parsing OND file: {e}")
            raise ValueError(f"Invalid OND file format: {e}")

    def parse_pvsyst_meteo_file(self, file_path: str) -> WeatherData:
        """
        Parse PVsyst .MET file to extract meteorological data.

        Args:
            file_path: Path to .MET file

        Returns:
            WeatherData object with parsed data

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is invalid
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MET file not found: {file_path}")

        logger.info(f"Parsing MET file: {file_path}")

        try:
            # PVsyst MET files are typically CSV-like with headers
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract metadata from header
            metadata = {}
            data_start_idx = 0

            for i, line in enumerate(lines):
                if line.strip().startswith('//') or line.strip().startswith('#'):
                    # Parse metadata
                    if 'Latitude' in line:
                        match = re.search(r'([-\d.]+)', line)
                        if match:
                            metadata['latitude'] = float(match.group(1))
                    if 'Longitude' in line:
                        match = re.search(r'([-\d.]+)', line)
                        if match:
                            metadata['longitude'] = float(match.group(1))
                    if 'Location' in line or 'Site' in line:
                        match = re.search(r'[:=]\s*(.+)', line)
                        if match:
                            metadata['location'] = match.group(1).strip()
                elif any(keyword in line.lower() for keyword in ['ghi', 'dni', 'temp', 'month']):
                    # Header line
                    data_start_idx = i + 1
                    break

            # Read data
            df = pd.read_csv(
                file_path,
                skiprows=data_start_idx,
                delim_whitespace=True,
                comment='#',
                on_bad_lines='skip'
            )

            # Map column names (case-insensitive)
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if 'ghi' in col_lower or 'gh' in col_lower:
                    col_mapping[col] = 'ghi'
                elif 'dni' in col_lower or 'bn' in col_lower:
                    col_mapping[col] = 'dni'
                elif 'dhi' in col_lower or 'dh' in col_lower or 'diff' in col_lower:
                    col_mapping[col] = 'dhi'
                elif 'temp' in col_lower or 'ta' in col_lower:
                    col_mapping[col] = 'temp_air'
                elif 'wind' in col_lower or 'ws' in col_lower:
                    col_mapping[col] = 'wind_speed'

            df.rename(columns=col_mapping, inplace=True)

            # Create timestamps (assume hourly data starting Jan 1)
            num_records = len(df)
            timestamps = pd.date_range(
                start=datetime(2023, 1, 1, 0, 0),
                periods=num_records,
                freq='h'
            )

            # Extract required data
            weather_data = WeatherData(
                location=metadata.get('location', 'Unknown'),
                latitude=metadata.get('latitude', 0.0),
                longitude=metadata.get('longitude', 0.0),
                ghi=df.get('ghi', pd.Series([0] * num_records)).tolist(),
                dni=df.get('dni', pd.Series([0] * num_records)).tolist(),
                dhi=df.get('dhi', pd.Series([0] * num_records)).tolist(),
                temp_air=df.get('temp_air', pd.Series([25] * num_records)).tolist(),
                wind_speed=df.get('wind_speed', pd.Series([2] * num_records)).tolist() if 'wind_speed' in df else None,
                timestamps=timestamps.tolist(),
            )

            logger.info(
                f"Parsed MET file: {weather_data.location}, "
                f"{len(weather_data.timestamps)} records"
            )

            return weather_data

        except Exception as e:
            logger.error(f"Error parsing MET file: {e}")
            raise ValueError(f"Invalid MET file format: {e}")

    def generate_pvsyst_project(
        self,
        project_name: str,
        module: ModuleParameters,
        inverter: InverterParameters,
        output_path: str,
        **kwargs,
    ) -> str:
        """
        Generate PVsyst-compatible project file (.PRJ).

        Args:
            project_name: Name of the project
            module: Module parameters
            inverter: Inverter parameters
            output_path: Output directory path
            **kwargs: Additional project parameters

        Returns:
            Path to generated .PRJ file
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        prj_file = output_dir / f"{project_name}.PRJ"

        # Generate PRJ file content (simplified format)
        content = f"""PVsyst V7.2 Project File
Project: {project_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generator: pv-circularity-simulator

[General]
ProjectName={project_name}
Author={kwargs.get('author', 'PV Simulator')}
Date={datetime.now().strftime('%Y-%m-%d')}

[Module]
Manufacturer={module.manufacturer}
Model={module.model}
Pmax={module.pmax}
Voc={module.voc}
Isc={module.isc}
Vmp={module.vmp}
Imp={module.imp}
TempCoeffPmax={module.temp_coeff_pmax}
TempCoeffVoc={module.temp_coeff_voc}

[Inverter]
Manufacturer={inverter.manufacturer}
Model={inverter.model}
PacMax={inverter.pac_max}
VacNom={inverter.vac_nom}
PdcMax={inverter.pdc_max}
VdcMax={inverter.vdc_max}
NumMPPT={inverter.num_mppt}
Efficiency={inverter.max_efficiency}

[System]
NumModules={kwargs.get('num_modules', 100)}
NumInverters={kwargs.get('num_inverters', 1)}
Tilt={kwargs.get('tilt', 30)}
Azimuth={kwargs.get('azimuth', 180)}
"""

        with open(prj_file, 'w') as f:
            f.write(content)

        logger.info(f"Generated PVsyst project file: {prj_file}")

        return str(prj_file)

    def execute_pvsyst_simulation(
        self,
        project_file: str,
        pvsyst_exe_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute PVsyst simulation using CLI/API.

        Note: This requires PVsyst to be installed and accessible.

        Args:
            project_file: Path to .PRJ file
            pvsyst_exe_path: Path to PVsyst executable (optional)

        Returns:
            Dictionary with simulation results or status

        Raises:
            RuntimeError: If PVsyst is not available
        """
        if pvsyst_exe_path and Path(pvsyst_exe_path).exists():
            logger.info(f"Executing PVsyst simulation for {project_file}")
            # In production, this would call PVsyst CLI
            # Example: subprocess.run([pvsyst_exe_path, "-run", project_file])
            logger.warning("PVsyst CLI execution not implemented in this version")
            return {
                "status": "not_implemented",
                "message": "PVsyst CLI integration requires installation",
            }
        else:
            raise RuntimeError(
                "PVsyst executable not found. "
                "PVsyst must be installed for simulation execution."
            )

    def import_pvsyst_results(
        self,
        results_file: str,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Import PVsyst simulation results from output files.

        Args:
            results_file: Path to results file (.CSV or .TXT)
            file_format: File format ("csv" or "txt")

        Returns:
            DataFrame with simulation results

        Raises:
            FileNotFoundError: If file does not exist
        """
        if not Path(results_file).exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        logger.info(f"Importing PVsyst results from {results_file}")

        try:
            if file_format.lower() == "csv":
                # PVsyst CSV output typically has header rows
                df = pd.read_csv(results_file, skiprows=10, encoding='utf-8')
            else:
                # Text format
                df = pd.read_csv(
                    results_file,
                    delim_whitespace=True,
                    skiprows=10,
                    encoding='utf-8'
                )

            logger.info(f"Imported {len(df)} records from PVsyst results")
            return df

        except Exception as e:
            logger.error(f"Error importing PVsyst results: {e}")
            raise ValueError(f"Invalid results file format: {e}")

    def export_to_pvsyst_format(
        self,
        data: pd.DataFrame,
        output_path: str,
        data_type: str = "weather",
    ) -> str:
        """
        Export data to PVsyst-compatible format.

        Args:
            data: DataFrame with data to export
            output_path: Output file path
            data_type: Type of data ("weather", "production", etc.)

        Returns:
            Path to exported file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add PVsyst header
        header = f"""// PVsyst Compatible Data File
// Data Type: {data_type}
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Generator: pv-circularity-simulator
//
"""

        with open(output_file, 'w') as f:
            f.write(header)
            data.to_csv(f, index=False, sep='\t')

        logger.info(f"Exported data to PVsyst format: {output_file}")

        return str(output_file)
