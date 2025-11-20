"""
Bifacial Module Modeling & Backside Irradiance

This module provides comprehensive modeling for bifacial photovoltaic modules,
including backside irradiance calculations, view factor modeling, bifacial gain
analysis, and performance simulation under various mounting configurations.

Author: PV Circularity Simulator
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Literal
from pydantic import BaseModel, Field, validator, root_validator
from dataclasses import dataclass
from enum import Enum
import warnings


# ============================================================================
# Enumerations and Constants
# ============================================================================

class AlbedoType(str, Enum):
    """Standard ground surface albedo types."""
    GRASS = "grass"
    CONCRETE = "concrete"
    WHITE_MEMBRANE = "white_membrane"
    SAND = "sand"
    SNOW = "snow"
    SOIL = "soil"
    ASPHALT = "asphalt"
    GRAVEL = "gravel"
    WATER = "water"


class MountingType(str, Enum):
    """Mounting structure types for bifacial modules."""
    FIXED_TILT = "fixed_tilt"
    SINGLE_AXIS_TRACKER = "single_axis_tracker"
    DUAL_AXIS_TRACKER = "dual_axis_tracker"
    VERTICAL = "vertical"
    EAST_WEST = "east_west"


class ViewFactorModel(str, Enum):
    """View factor calculation models."""
    PEREZ = "perez"
    DURUSOY = "durusoy"
    SIMPLE = "simple"


# Standard albedo values
ALBEDO_VALUES = {
    AlbedoType.GRASS: 0.20,
    AlbedoType.CONCRETE: 0.30,
    AlbedoType.WHITE_MEMBRANE: 0.70,
    AlbedoType.SAND: 0.40,
    AlbedoType.SNOW: 0.80,
    AlbedoType.SOIL: 0.17,
    AlbedoType.ASPHALT: 0.12,
    AlbedoType.GRAVEL: 0.25,
    AlbedoType.WATER: 0.06,
}


# ============================================================================
# Pydantic Models
# ============================================================================

class TMY(BaseModel):
    """Typical Meteorological Year data model."""
    ghi: Union[pd.Series, List[float]] = Field(..., description="Global Horizontal Irradiance (W/m²)")
    dni: Union[pd.Series, List[float]] = Field(..., description="Direct Normal Irradiance (W/m²)")
    dhi: Union[pd.Series, List[float]] = Field(..., description="Diffuse Horizontal Irradiance (W/m²)")
    temp_air: Union[pd.Series, List[float]] = Field(..., description="Air temperature (°C)")
    wind_speed: Union[pd.Series, List[float]] = Field(..., description="Wind speed (m/s)")
    solar_zenith: Optional[Union[pd.Series, List[float]]] = Field(None, description="Solar zenith angle (degrees)")
    solar_azimuth: Optional[Union[pd.Series, List[float]]] = Field(None, description="Solar azimuth angle (degrees)")

    class Config:
        arbitrary_types_allowed = True


class MountingStructure(BaseModel):
    """Mounting structure configuration for bifacial modules."""
    mounting_type: MountingType = Field(..., description="Type of mounting structure")
    tilt: float = Field(..., ge=0, le=90, description="Tilt angle from horizontal (degrees)")
    azimuth: float = Field(180.0, ge=0, le=360, description="Azimuth angle (degrees, 180=south)")
    clearance_height: float = Field(..., gt=0, le=10, description="Ground clearance height (meters)")
    row_spacing: Optional[float] = Field(None, gt=0, description="Row-to-row spacing (meters)")
    row_width: Optional[float] = Field(None, gt=0, description="Row width (meters)")
    n_rows: int = Field(1, ge=1, description="Number of module rows")
    tracker_max_angle: Optional[float] = Field(60.0, ge=0, le=90, description="Maximum tracker rotation angle (degrees)")

    @validator('row_spacing', 'row_width')
    def validate_spacing(cls, v, values):
        """Validate row spacing is provided for multi-row systems."""
        if values.get('n_rows', 1) > 1 and v is None:
            raise ValueError("row_spacing and row_width required for multi-row systems")
        return v


class BifacialModuleParams(BaseModel):
    """Bifacial module parameters and specifications."""
    bifaciality: float = Field(..., ge=0.5, le=1.0, description="Bifaciality coefficient (rear/front efficiency)")
    front_efficiency: float = Field(..., ge=0.05, le=0.30, description="Front cell efficiency (fraction)")
    rear_efficiency: Optional[float] = Field(None, ge=0.05, le=0.30, description="Rear cell efficiency (fraction)")
    glass_transmission_front: float = Field(0.91, ge=0.7, le=0.98, description="Front glass transmission")
    glass_transmission_rear: float = Field(0.88, ge=0.7, le=0.98, description="Rear glass transmission")
    encapsulant_absorption_rear: float = Field(0.03, ge=0, le=0.15, description="Rear encapsulant absorption")
    temp_coeff_pmax: float = Field(-0.0037, ge=-0.006, le=0, description="Temperature coefficient of Pmax (%/°C)")
    temp_coeff_bifacial: float = Field(0.90, ge=0.8, le=1.0, description="Bifacial temperature coefficient factor")
    module_width: float = Field(1.0, gt=0, le=3, description="Module width (meters)")
    module_length: float = Field(2.0, gt=0, le=3, description="Module length (meters)")

    @validator('rear_efficiency', always=True)
    def set_rear_efficiency(cls, v, values):
        """Calculate rear efficiency from bifaciality if not provided."""
        if v is None and 'bifaciality' in values and 'front_efficiency' in values:
            return values['bifaciality'] * values['front_efficiency']
        return v


class GroundSurface(BaseModel):
    """Ground surface properties for reflection modeling."""
    albedo: float = Field(..., ge=0, le=1, description="Ground albedo (reflectance)")
    albedo_type: Optional[AlbedoType] = Field(None, description="Standard albedo type")
    seasonal_variation: bool = Field(False, description="Model seasonal albedo variation")
    snow_cover_threshold: float = Field(0.0, ge=0, le=1, description="Snow cover fraction threshold")

    @root_validator
    def validate_albedo_or_type(cls, values):
        """Set albedo from type if provided."""
        if values.get('albedo_type') is not None:
            values['albedo'] = ALBEDO_VALUES[values['albedo_type']]
        return values


class BifacialSystemConfig(BaseModel):
    """Complete bifacial PV system configuration."""
    module: BifacialModuleParams
    structure: MountingStructure
    ground: GroundSurface
    view_factor_model: ViewFactorModel = Field(ViewFactorModel.PEREZ, description="View factor calculation model")
    enable_mismatch_losses: bool = Field(True, description="Enable mismatch loss calculations")
    enable_soiling: bool = Field(False, description="Enable soiling impact modeling")
    soiling_factor_front: float = Field(0.98, ge=0.8, le=1.0, description="Front soiling factor")
    soiling_factor_rear: float = Field(0.95, ge=0.8, le=1.0, description="Rear soiling factor")
    location_latitude: float = Field(..., ge=-90, le=90, description="Site latitude (degrees)")
    location_longitude: float = Field(..., ge=-180, le=180, description="Site longitude (degrees)")


# ============================================================================
# View Factor Calculations
# ============================================================================

class ViewFactorCalculator:
    """
    Calculate view factors for bifacial module backside irradiance.

    Implements multiple view factor models including Perez and Durusoy methods
    for accurate backside irradiance estimation.
    """

    @staticmethod
    def perez_view_factor(
        tilt: float,
        clearance: float,
        row_spacing: Optional[float] = None,
        row_width: Optional[float] = None,
        row_number: int = 1,
        total_rows: int = 1
    ) -> Dict[str, float]:
        """
        Calculate view factors using the Perez model.

        Based on: Perez et al. (2012) "A Practical Method for the Design of
        Bifacial PV Systems"

        Args:
            tilt: Module tilt angle from horizontal (degrees)
            clearance: Ground clearance height (meters)
            row_spacing: Distance between row centers (meters)
            row_width: Width of each row (meters)
            row_number: Current row number (1-indexed)
            total_rows: Total number of rows

        Returns:
            Dictionary with view factors:
                - f_gnd_beam: Ground view factor for beam irradiance
                - f_gnd_diff: Ground view factor for diffuse irradiance
                - f_sky: Sky view factor
                - f_row: View factor to adjacent rows (for shading)
        """
        tilt_rad = np.radians(tilt)

        # Sky view factor (basic geometric relationship)
        f_sky = (1 + np.cos(tilt_rad)) / 2

        # For single row or infinite spacing
        if row_spacing is None or total_rows == 1:
            f_gnd_diff = (1 - np.cos(tilt_rad)) / 2
            f_gnd_beam = f_gnd_diff
            f_row = 0.0
            return {
                'f_gnd_beam': f_gnd_beam,
                'f_gnd_diff': f_gnd_diff,
                'f_sky': f_sky,
                'f_row': f_row
            }

        # Multi-row configuration
        # Calculate effective dimensions
        module_height = row_width * np.sin(tilt_rad)
        module_projection = row_width * np.cos(tilt_rad)

        # Ground coverage ratio
        gcr = row_width / row_spacing if row_spacing > 0 else 0.0
        gcr = min(gcr, 0.99)  # Prevent division by zero

        # Effective ground view angle considering row spacing
        # Distance from module edge to next row
        ground_gap = row_spacing - module_projection

        # Angle to ground from module edge
        if ground_gap > 0:
            ground_angle = np.arctan(clearance / ground_gap)
        else:
            ground_angle = np.pi / 2

        # Perez ground view factor with row spacing correction
        # For beam irradiance (depends on sun position, approximated here)
        f_gnd_beam = (1 - np.cos(tilt_rad)) / 2 * (1 - gcr * 0.5)

        # For diffuse irradiance
        vf_base = (1 - np.cos(tilt_rad)) / 2

        # Correction for row-to-row shading
        shading_correction = 1 - (gcr * np.sin(tilt_rad) * 0.5)
        f_gnd_diff = vf_base * shading_correction

        # Inter-row view factor (for reflected light from adjacent modules)
        f_row = gcr * np.sin(tilt_rad) * 0.3  # Empirical factor

        # Edge row enhancement (first and last rows see more ground)
        if row_number == 1 or row_number == total_rows:
            edge_enhancement = 1.15
            f_gnd_beam *= edge_enhancement
            f_gnd_diff *= edge_enhancement

        return {
            'f_gnd_beam': min(f_gnd_beam, 1.0),
            'f_gnd_diff': min(f_gnd_diff, 1.0),
            'f_sky': f_sky,
            'f_row': min(f_row, 1.0)
        }

    @staticmethod
    def durusoy_view_factor(
        tilt: float,
        clearance: float,
        row_spacing: Optional[float] = None,
        row_width: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate view factors using the Durusoy model.

        Based on: Durusoy et al. (2020) "Solar irradiance on the rear surface
        of bifacial solar modules"

        Args:
            tilt: Module tilt angle from horizontal (degrees)
            clearance: Ground clearance height (meters)
            row_spacing: Distance between row centers (meters)
            row_width: Width of each row (meters)

        Returns:
            Dictionary with view factors
        """
        tilt_rad = np.radians(tilt)

        # Sky view factor
        f_sky = (1 + np.cos(tilt_rad)) / 2

        # Single row case
        if row_spacing is None or row_width is None:
            f_gnd = (1 - np.cos(tilt_rad)) / 2
            return {
                'f_gnd_beam': f_gnd,
                'f_gnd_diff': f_gnd,
                'f_sky': f_sky,
                'f_row': 0.0
            }

        # Durusoy geometric model
        L = row_width  # Module width
        h = clearance  # Ground clearance
        p = row_spacing  # Pitch
        beta = tilt_rad  # Tilt angle

        # Effective ground viewing dimensions
        L_rear = L * np.cos(beta)  # Horizontal projection
        h_eff = h + L * np.sin(beta) / 2  # Effective height to module center

        # Distance to visible ground
        d1 = h / np.tan(beta) if beta > 0 else 0
        d2 = p - L_rear

        # View factor to ground (Durusoy formulation)
        if d2 > 0:
            # Angle subtended by visible ground
            theta1 = np.arctan(h_eff / d1) if d1 > 0 else np.pi/2
            theta2 = np.arctan(h_eff / (d1 + d2))

            # Ground view factor
            f_gnd = (np.sin(theta1) - np.sin(theta2)) * (1 - np.cos(beta)) / 2
        else:
            f_gnd = (1 - np.cos(beta)) / 2

        # Inter-row view factor
        f_row = (1 - f_sky - f_gnd) * 0.5

        return {
            'f_gnd_beam': f_gnd,
            'f_gnd_diff': f_gnd,
            'f_sky': f_sky,
            'f_row': max(f_row, 0.0)
        }

    @staticmethod
    def simple_view_factor(tilt: float) -> Dict[str, float]:
        """
        Simple isotropic view factor calculation.

        Args:
            tilt: Module tilt angle from horizontal (degrees)

        Returns:
            Dictionary with view factors
        """
        tilt_rad = np.radians(tilt)

        f_sky = (1 + np.cos(tilt_rad)) / 2
        f_gnd = (1 - np.cos(tilt_rad)) / 2

        return {
            'f_gnd_beam': f_gnd,
            'f_gnd_diff': f_gnd,
            'f_sky': f_sky,
            'f_row': 0.0
        }


# ============================================================================
# Bifacial Module Model
# ============================================================================

class BifacialModuleModel:
    """
    Comprehensive bifacial PV module modeling.

    This class provides complete modeling capabilities for bifacial photovoltaic
    modules including:
    - Backside irradiance calculation with view factors
    - Bifacial gain analysis
    - Performance simulation under various conditions
    - Mounting structure effects
    - Advanced loss mechanisms (mismatch, temperature, soiling)

    Example:
        >>> from bifacial_model import BifacialModuleModel, BifacialSystemConfig
        >>> model = BifacialModuleModel(config)
        >>> back_irr = model.calculate_backside_irradiance(0.25, 30, 1.0)
        >>> gain = model.calculate_bifacial_gain(1000, back_irr, 0.70)
    """

    def __init__(self, config: Optional[BifacialSystemConfig] = None):
        """
        Initialize the bifacial module model.

        Args:
            config: System configuration (optional, can be set later)
        """
        self.config = config
        self.view_factor_calc = ViewFactorCalculator()

    def calculate_backside_irradiance(
        self,
        ground_albedo: float,
        tilt: float,
        clearance: float,
        front_poa_global: float = 1000.0,
        front_poa_beam: float = 700.0,
        front_poa_diffuse: float = 300.0,
        dhi: float = 100.0,
        row_spacing: Optional[float] = None,
        row_width: Optional[float] = None,
        row_number: int = 1,
        total_rows: int = 1,
        view_factor_model: ViewFactorModel = ViewFactorModel.PEREZ
    ) -> float:
        """
        Calculate backside irradiance on bifacial module.

        Includes contributions from:
        - Ground-reflected beam irradiance
        - Ground-reflected diffuse irradiance
        - Sky diffuse irradiance
        - Inter-row reflections (for multi-row systems)

        Args:
            ground_albedo: Ground surface albedo (0-1)
            tilt: Module tilt angle from horizontal (degrees)
            clearance: Ground clearance height (meters)
            front_poa_global: Front plane-of-array global irradiance (W/m²)
            front_poa_beam: Front plane-of-array beam irradiance (W/m²)
            front_poa_diffuse: Front plane-of-array diffuse irradiance (W/m²)
            dhi: Diffuse horizontal irradiance (W/m²)
            row_spacing: Row-to-row spacing (meters)
            row_width: Module row width (meters)
            row_number: Current row number (1-indexed)
            total_rows: Total number of rows
            view_factor_model: View factor calculation model

        Returns:
            Backside irradiance (W/m²)
        """
        # Calculate view factors based on selected model
        if view_factor_model == ViewFactorModel.PEREZ:
            vf = self.view_factor_calc.perez_view_factor(
                tilt, clearance, row_spacing, row_width, row_number, total_rows
            )
        elif view_factor_model == ViewFactorModel.DURUSOY:
            vf = self.view_factor_calc.durusoy_view_factor(
                tilt, clearance, row_spacing, row_width
            )
        else:
            vf = self.view_factor_calc.simple_view_factor(tilt)

        # Ground-reflected beam component
        # Beam irradiance hitting ground is reflected
        ground_reflected_beam = front_poa_beam * ground_albedo * vf['f_gnd_beam']

        # Ground-reflected diffuse component
        ground_reflected_diffuse = front_poa_diffuse * ground_albedo * vf['f_gnd_diff']

        # Sky diffuse component (rear side can see sky)
        sky_diffuse = dhi * vf['f_sky']

        # Inter-row reflection (simplified)
        # Adjacent modules can reflect light to rear side
        interrow_reflection = front_poa_global * 0.10 * vf['f_row']  # Assume 10% module reflectance

        # Total backside irradiance
        backside_irradiance = (
            ground_reflected_beam +
            ground_reflected_diffuse +
            sky_diffuse +
            interrow_reflection
        )

        return max(backside_irradiance, 0.0)

    def calculate_bifacial_gain(
        self,
        front_irr: float,
        back_irr: float,
        bifaciality: float
    ) -> float:
        """
        Calculate bifacial gain factor.

        Bifacial gain represents the additional energy production from the
        rear side compared to a monofacial module.

        Args:
            front_irr: Front-side irradiance (W/m²)
            back_irr: Back-side irradiance (W/m²)
            bifaciality: Module bifaciality coefficient (0-1)

        Returns:
            Bifacial gain (fraction, e.g., 0.15 = 15% gain)
        """
        if front_irr <= 0:
            return 0.0

        # Bifacial gain = (rear contribution) / (front contribution)
        rear_contribution = back_irr * bifaciality
        bifacial_gain = rear_contribution / front_irr

        return bifacial_gain

    def model_view_factors(
        self,
        structure: MountingStructure,
        view_factor_model: ViewFactorModel = ViewFactorModel.PEREZ
    ) -> Dict:
        """
        Calculate view factors for a given mounting structure.

        Args:
            structure: Mounting structure configuration
            view_factor_model: View factor calculation model to use

        Returns:
            Dictionary containing view factors for each row and statistics
        """
        results = {
            'model': view_factor_model.value,
            'rows': [],
            'average_f_gnd_beam': 0.0,
            'average_f_gnd_diff': 0.0,
            'average_f_sky': 0.0,
            'average_f_row': 0.0
        }

        # Calculate view factors for each row
        for row_num in range(1, structure.n_rows + 1):
            if view_factor_model == ViewFactorModel.PEREZ:
                vf = self.view_factor_calc.perez_view_factor(
                    structure.tilt,
                    structure.clearance_height,
                    structure.row_spacing,
                    structure.row_width,
                    row_num,
                    structure.n_rows
                )
            elif view_factor_model == ViewFactorModel.DURUSOY:
                vf = self.view_factor_calc.durusoy_view_factor(
                    structure.tilt,
                    structure.clearance_height,
                    structure.row_spacing,
                    structure.row_width
                )
            else:
                vf = self.view_factor_calc.simple_view_factor(structure.tilt)

            vf['row_number'] = row_num
            results['rows'].append(vf)

        # Calculate averages
        if results['rows']:
            results['average_f_gnd_beam'] = np.mean([r['f_gnd_beam'] for r in results['rows']])
            results['average_f_gnd_diff'] = np.mean([r['f_gnd_diff'] for r in results['rows']])
            results['average_f_sky'] = np.mean([r['f_sky'] for r in results['rows']])
            results['average_f_row'] = np.mean([r['f_row'] for r in results['rows']])

        return results

    def calculate_effective_irradiance(
        self,
        front: float,
        back: float,
        bifaciality: float,
        glass_transmission_front: float = 0.91,
        glass_transmission_rear: float = 0.88,
        encapsulant_absorption_rear: float = 0.03
    ) -> float:
        """
        Calculate effective irradiance accounting for optical losses.

        The effective irradiance is the total irradiance reaching the cells
        after accounting for glass transmission and encapsulant absorption.

        Args:
            front: Front-side plane-of-array irradiance (W/m²)
            back: Back-side irradiance (W/m²)
            bifaciality: Module bifaciality coefficient
            glass_transmission_front: Front glass transmission (fraction)
            glass_transmission_rear: Rear glass transmission (fraction)
            encapsulant_absorption_rear: Rear encapsulant absorption (fraction)

        Returns:
            Effective irradiance at cell level (W/m²)
        """
        # Front contribution with glass losses
        front_effective = front * glass_transmission_front

        # Rear contribution with glass and encapsulant losses
        rear_transmission = glass_transmission_rear * (1 - encapsulant_absorption_rear)
        rear_effective = back * rear_transmission * bifaciality

        # Total effective irradiance (monofacial equivalent)
        effective_irradiance = front_effective + rear_effective

        return effective_irradiance

    def calculate_mismatch_losses(
        self,
        back_irr_distribution: np.ndarray,
        front_irr: float
    ) -> float:
        """
        Calculate mismatch losses from non-uniform rear irradiance.

        Non-uniform backside irradiance can cause current mismatch between
        cells/strings, leading to power losses.

        Args:
            back_irr_distribution: Array of rear irradiance values across module (W/m²)
            front_irr: Front irradiance (W/m²)

        Returns:
            Mismatch loss factor (fraction)
        """
        if len(back_irr_distribution) == 0:
            return 0.0

        # Calculate coefficient of variation
        mean_back = np.mean(back_irr_distribution)
        std_back = np.std(back_irr_distribution)

        if mean_back == 0:
            return 0.0

        cv = std_back / mean_back

        # Empirical relationship: mismatch loss proportional to CV²
        # Typical values: CV=0.1 -> 1% loss, CV=0.2 -> 4% loss
        mismatch_loss = min(cv**2 * 0.5, 0.15)  # Cap at 15%

        return mismatch_loss

    def calculate_temperature_effect(
        self,
        front_irr: float,
        back_irr: float,
        ambient_temp: float,
        wind_speed: float,
        temp_coeff: float = -0.0037,
        temp_coeff_bifacial: float = 0.90,
        noct: float = 45.0
    ) -> Tuple[float, float]:
        """
        Calculate cell temperature and temperature coefficient for bifacial module.

        Bifacial modules can be cooler than monofacial due to airflow on both sides.

        Args:
            front_irr: Front irradiance (W/m²)
            back_irr: Rear irradiance (W/m²)
            ambient_temp: Ambient air temperature (°C)
            wind_speed: Wind speed (m/s)
            temp_coeff: Temperature coefficient of Pmax (%/°C)
            temp_coeff_bifacial: Bifacial temperature coefficient factor
            noct: Nominal Operating Cell Temperature (°C)

        Returns:
            Tuple of (cell_temperature, temperature_loss_factor)
        """
        # Total irradiance absorbed
        total_irr = front_irr + back_irr * 0.5  # Rear side contributes less to heating

        # Wind speed correction factor
        wind_factor = 1.0 - 0.05 * min(wind_speed, 10.0)  # Higher wind = better cooling

        # Cell temperature estimation (modified Faiman model)
        # Bifacial modules run cooler due to better airflow
        delta_t = (noct - 20) * (total_irr / 800) * wind_factor * temp_coeff_bifacial
        cell_temp = ambient_temp + delta_t

        # Temperature loss factor
        temp_diff = cell_temp - 25.0  # STC is 25°C
        temp_loss_factor = 1 + (temp_coeff * temp_diff)

        return cell_temp, temp_loss_factor

    def calculate_soiling_impact(
        self,
        front_soiling: float,
        rear_soiling: float,
        front_irr: float,
        back_irr: float,
        bifaciality: float
    ) -> Tuple[float, float]:
        """
        Calculate soiling impact on bifacial module performance.

        Args:
            front_soiling: Front soiling factor (0-1, 1=clean)
            rear_soiling: Rear soiling factor (0-1, 1=clean)
            front_irr: Front irradiance before soiling (W/m²)
            back_irr: Rear irradiance before soiling (W/m²)
            bifaciality: Module bifaciality coefficient

        Returns:
            Tuple of (effective_front_irr, effective_back_irr) after soiling
        """
        effective_front = front_irr * front_soiling
        effective_back = back_irr * rear_soiling

        return effective_front, effective_back

    def optimize_row_spacing(
        self,
        module_width: float,
        tilt: float,
        ground_albedo: float,
        clearance: float,
        latitude: float,
        max_gcr: float = 0.5,
        min_gcr: float = 0.2,
        n_points: int = 20
    ) -> Dict:
        """
        Optimize row-to-row spacing for maximum bifacial gain.

        Args:
            module_width: Module width (meters)
            tilt: Module tilt angle (degrees)
            ground_albedo: Ground albedo
            clearance: Ground clearance (meters)
            latitude: Site latitude (degrees)
            max_gcr: Maximum ground coverage ratio
            min_gcr: Minimum ground coverage ratio
            n_points: Number of spacing points to evaluate

        Returns:
            Dictionary with optimization results
        """
        gcr_values = np.linspace(min_gcr, max_gcr, n_points)
        results = []

        for gcr in gcr_values:
            row_spacing = module_width / gcr

            # Calculate backside irradiance at typical conditions
            back_irr = self.calculate_backside_irradiance(
                ground_albedo=ground_albedo,
                tilt=tilt,
                clearance=clearance,
                front_poa_global=1000.0,
                front_poa_beam=700.0,
                front_poa_diffuse=300.0,
                dhi=100.0,
                row_spacing=row_spacing,
                row_width=module_width,
                total_rows=5  # Assume interior row
            )

            # Calculate energy density (normalized by land area)
            bifacial_gain = self.calculate_bifacial_gain(1000.0, back_irr, 0.70)
            energy_per_module = 1000.0 * (1 + bifacial_gain)
            energy_density = energy_per_module * gcr  # Energy per unit land area

            results.append({
                'gcr': gcr,
                'row_spacing': row_spacing,
                'back_irradiance': back_irr,
                'bifacial_gain': bifacial_gain,
                'energy_density': energy_density
            })

        # Find optimal GCR
        optimal_idx = np.argmax([r['energy_density'] for r in results])
        optimal = results[optimal_idx]

        return {
            'optimal_gcr': optimal['gcr'],
            'optimal_spacing': optimal['row_spacing'],
            'optimal_bifacial_gain': optimal['bifacial_gain'],
            'optimization_curve': pd.DataFrame(results)
        }

    def simulate_bifacial_performance(
        self,
        system: Dict,
        weather: TMY,
        detailed_output: bool = False
    ) -> pd.DataFrame:
        """
        Simulate bifacial PV system performance over time.

        Args:
            system: System configuration dictionary with keys:
                - module: BifacialModuleParams
                - structure: MountingStructure
                - ground: GroundSurface
                - latitude: float
                - longitude: float
            weather: TMY weather data
            detailed_output: Include detailed intermediate calculations

        Returns:
            DataFrame with time-series performance data
        """
        # Extract configuration
        module = system['module'] if isinstance(system['module'], BifacialModuleParams) else BifacialModuleParams(**system['module'])
        structure = system['structure'] if isinstance(system['structure'], MountingStructure) else MountingStructure(**system['structure'])
        ground = system['ground'] if isinstance(system['ground'], GroundSurface) else GroundSurface(**system['ground'])

        # Convert TMY data to arrays
        ghi = np.array(weather.ghi if isinstance(weather.ghi, list) else weather.ghi.values)
        dni = np.array(weather.dni if isinstance(weather.dni, list) else weather.dni.values)
        dhi = np.array(weather.dhi if isinstance(weather.dhi, list) else weather.dhi.values)
        temp_air = np.array(weather.temp_air if isinstance(weather.temp_air, list) else weather.temp_air.values)
        wind_speed = np.array(weather.wind_speed if isinstance(weather.wind_speed, list) else weather.wind_speed.values)

        n_timesteps = len(ghi)

        # Initialize results
        results = {
            'timestamp': list(range(n_timesteps)),
            'front_poa_global': np.zeros(n_timesteps),
            'back_irradiance': np.zeros(n_timesteps),
            'effective_irradiance': np.zeros(n_timesteps),
            'bifacial_gain': np.zeros(n_timesteps),
            'cell_temperature': np.zeros(n_timesteps),
            'power_output': np.zeros(n_timesteps),
            'bifacial_power_gain': np.zeros(n_timesteps),
        }

        if detailed_output:
            results.update({
                'front_poa_beam': np.zeros(n_timesteps),
                'front_poa_diffuse': np.zeros(n_timesteps),
                'temp_loss_factor': np.zeros(n_timesteps),
                'mismatch_loss': np.zeros(n_timesteps),
            })

        # Simulate each timestep
        for i in range(n_timesteps):
            # Simple POA calculation (would use pvlib in production)
            # Assume optimal tracking or fixed tilt
            front_poa_global = ghi[i] * 1.1  # Simplified: assume some concentration
            front_poa_beam = dni[i] * 0.8
            front_poa_diffuse = dhi[i] * 1.3

            # Calculate backside irradiance
            back_irr = self.calculate_backside_irradiance(
                ground_albedo=ground.albedo,
                tilt=structure.tilt,
                clearance=structure.clearance_height,
                front_poa_global=front_poa_global,
                front_poa_beam=front_poa_beam,
                front_poa_diffuse=front_poa_diffuse,
                dhi=dhi[i],
                row_spacing=structure.row_spacing,
                row_width=structure.row_width,
                total_rows=structure.n_rows
            )

            # Calculate effective irradiance
            eff_irr = self.calculate_effective_irradiance(
                front=front_poa_global,
                back=back_irr,
                bifaciality=module.bifaciality,
                glass_transmission_front=module.glass_transmission_front,
                glass_transmission_rear=module.glass_transmission_rear,
                encapsulant_absorption_rear=module.encapsulant_absorption_rear
            )

            # Calculate bifacial gain
            bifacial_gain = self.calculate_bifacial_gain(
                front_poa_global, back_irr, module.bifaciality
            )

            # Temperature effects
            cell_temp, temp_loss = self.calculate_temperature_effect(
                front_irr=front_poa_global,
                back_irr=back_irr,
                ambient_temp=temp_air[i],
                wind_speed=wind_speed[i],
                temp_coeff=module.temp_coeff_pmax,
                temp_coeff_bifacial=module.temp_coeff_bifacial
            )

            # Power output (simplified)
            stc_power = 400.0  # Watts (typical module)
            power = stc_power * (eff_irr / 1000.0) * temp_loss

            # Monofacial equivalent for comparison
            mono_power = stc_power * (front_poa_global / 1000.0) * temp_loss

            # Store results
            results['front_poa_global'][i] = front_poa_global
            results['back_irradiance'][i] = back_irr
            results['effective_irradiance'][i] = eff_irr
            results['bifacial_gain'][i] = bifacial_gain
            results['cell_temperature'][i] = cell_temp
            results['power_output'][i] = power
            results['bifacial_power_gain'][i] = (power - mono_power) / mono_power if mono_power > 0 else 0

            if detailed_output:
                results['front_poa_beam'][i] = front_poa_beam
                results['front_poa_diffuse'][i] = front_poa_diffuse
                results['temp_loss_factor'][i] = temp_loss
                results['mismatch_loss'][i] = 0.02  # Placeholder

        return pd.DataFrame(results)


# ============================================================================
# Utility Functions
# ============================================================================

def get_albedo_seasonal_variation(
    base_albedo: float,
    month: int,
    snow_cover: bool = False
) -> float:
    """
    Get seasonally-adjusted albedo value.

    Args:
        base_albedo: Base albedo value
        month: Month number (1-12)
        snow_cover: Whether snow cover is present

    Returns:
        Adjusted albedo value
    """
    if snow_cover:
        return ALBEDO_VALUES[AlbedoType.SNOW]

    # Vegetation seasonal variation
    if base_albedo == ALBEDO_VALUES[AlbedoType.GRASS]:
        # Grass is greener (lower albedo) in spring/summer
        seasonal_factor = {
            1: 1.1, 2: 1.1, 3: 1.0, 4: 0.95, 5: 0.9, 6: 0.9,
            7: 0.95, 8: 1.0, 9: 1.0, 10: 1.05, 11: 1.1, 12: 1.1
        }
        return base_albedo * seasonal_factor.get(month, 1.0)

    return base_albedo


def calculate_gcr(row_width: float, row_spacing: float) -> float:
    """
    Calculate ground coverage ratio.

    Args:
        row_width: Module row width (meters)
        row_spacing: Row-to-row spacing (meters)

    Returns:
        Ground coverage ratio (0-1)
    """
    if row_spacing <= 0:
        raise ValueError("Row spacing must be positive")

    return min(row_width / row_spacing, 1.0)


def validate_bifacial_system(config: BifacialSystemConfig) -> List[str]:
    """
    Validate bifacial system configuration and return warnings.

    Args:
        config: System configuration to validate

    Returns:
        List of warning messages
    """
    warnings_list = []

    # Check clearance height
    if config.structure.clearance_height < 0.5:
        warnings_list.append(
            f"Low clearance height ({config.structure.clearance_height}m) may reduce bifacial gain. "
            "Consider increasing to 1.0m or higher."
        )

    # Check GCR for fixed tilt
    if config.structure.mounting_type == MountingType.FIXED_TILT:
        if config.structure.row_spacing and config.structure.row_width:
            gcr = calculate_gcr(config.structure.row_width, config.structure.row_spacing)
            if gcr > 0.5:
                warnings_list.append(
                    f"High GCR ({gcr:.2f}) may cause significant row-to-row shading. "
                    "Consider reducing GCR to 0.4-0.5 for optimal bifacial performance."
                )

    # Check bifaciality factor
    if config.module.bifaciality < 0.65:
        warnings_list.append(
            f"Low bifaciality factor ({config.module.bifaciality:.2f}). "
            "Modern bifacial modules typically have bifaciality > 0.70."
        )

    # Check albedo
    if config.ground.albedo < 0.15:
        warnings_list.append(
            f"Low ground albedo ({config.ground.albedo:.2f}) limits bifacial gain. "
            "Consider white membrane or other high-albedo ground cover."
        )

    return warnings_list


# ============================================================================
# Example Usage and Testing
# ============================================================================

def create_example_system() -> BifacialSystemConfig:
    """Create an example bifacial system configuration."""

    module = BifacialModuleParams(
        bifaciality=0.70,
        front_efficiency=0.21,
        glass_transmission_front=0.91,
        glass_transmission_rear=0.88,
        encapsulant_absorption_rear=0.03,
        temp_coeff_pmax=-0.0037,
        temp_coeff_bifacial=0.90,
        module_width=1.1,
        module_length=2.3
    )

    structure = MountingStructure(
        mounting_type=MountingType.FIXED_TILT,
        tilt=30.0,
        azimuth=180.0,
        clearance_height=1.0,
        row_spacing=4.0,
        row_width=1.1,
        n_rows=10
    )

    ground = GroundSurface(
        albedo=0.25,
        albedo_type=AlbedoType.GRASS,
        seasonal_variation=False
    )

    config = BifacialSystemConfig(
        module=module,
        structure=structure,
        ground=ground,
        view_factor_model=ViewFactorModel.PEREZ,
        location_latitude=35.0,
        location_longitude=-106.0
    )

    return config


if __name__ == "__main__":
    # Example usage
    print("Bifacial Module Modeling Example")
    print("=" * 60)

    # Create example configuration
    config = create_example_system()

    # Validate configuration
    warnings_list = validate_bifacial_system(config)
    if warnings_list:
        print("\nConfiguration Warnings:")
        for warning in warnings_list:
            print(f"  ⚠ {warning}")

    # Initialize model
    model = BifacialModuleModel(config)

    # Calculate backside irradiance
    print("\n1. Backside Irradiance Calculation")
    print("-" * 60)
    back_irr = model.calculate_backside_irradiance(
        ground_albedo=0.25,
        tilt=30.0,
        clearance=1.0,
        front_poa_global=1000.0,
        front_poa_beam=700.0,
        front_poa_diffuse=300.0,
        dhi=100.0,
        row_spacing=4.0,
        row_width=1.1,
        total_rows=10
    )
    print(f"Front POA: 1000 W/m²")
    print(f"Backside irradiance: {back_irr:.1f} W/m²")

    # Calculate bifacial gain
    print("\n2. Bifacial Gain Calculation")
    print("-" * 60)
    gain = model.calculate_bifacial_gain(1000.0, back_irr, 0.70)
    print(f"Bifacial gain: {gain*100:.1f}%")

    # Model view factors
    print("\n3. View Factor Analysis")
    print("-" * 60)
    vf_results = model.model_view_factors(config.structure, ViewFactorModel.PEREZ)
    print(f"Average ground view factor (beam): {vf_results['average_f_gnd_beam']:.3f}")
    print(f"Average ground view factor (diffuse): {vf_results['average_f_gnd_diff']:.3f}")
    print(f"Average sky view factor: {vf_results['average_f_sky']:.3f}")

    # Calculate effective irradiance
    print("\n4. Effective Irradiance")
    print("-" * 60)
    eff_irr = model.calculate_effective_irradiance(
        front=1000.0,
        back=back_irr,
        bifaciality=0.70
    )
    print(f"Effective irradiance: {eff_irr:.1f} W/m²")
    print(f"Monofacial equivalent: {1000.0 * 0.91:.1f} W/m²")
    print(f"Bifacial enhancement: {(eff_irr / (1000.0 * 0.91) - 1) * 100:.1f}%")

    # Optimize row spacing
    print("\n5. Row Spacing Optimization")
    print("-" * 60)
    opt_results = model.optimize_row_spacing(
        module_width=1.1,
        tilt=30.0,
        ground_albedo=0.25,
        clearance=1.0,
        latitude=35.0
    )
    print(f"Optimal GCR: {opt_results['optimal_gcr']:.2f}")
    print(f"Optimal spacing: {opt_results['optimal_spacing']:.2f} m")
    print(f"Bifacial gain at optimal: {opt_results['optimal_bifacial_gain']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
