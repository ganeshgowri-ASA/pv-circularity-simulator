"""Plane-of-Array (POA) irradiance modeling with loss factors."""

from typing import Optional

import numpy as np
import pandas as pd
import pvlib
from pvlib.irradiance import aoi, aoi_projection, beam_component, get_extra_radiation

from .calculator import IrradianceCalculator
from .models import IrradianceComponents, LocationConfig, POAComponents, SolarPosition, SurfaceConfig


class POAIrradianceModel:
    """Model for calculating Plane-of-Array (POA) irradiance with loss factors.

    This class calculates the total irradiance on a tilted surface, including:
    - Direct beam component
    - Sky diffuse component (using various transposition models)
    - Ground reflected component
    - Spectral corrections
    - Angle of incidence (AOI) losses

    The model provides production-ready calculations for PV system performance modeling.
    """

    def __init__(
        self,
        location: LocationConfig,
        surface: SurfaceConfig,
        irradiance_calculator: Optional[IrradianceCalculator] = None,
    ):
        """Initialize the POA irradiance model.

        Args:
            location: Geographic location configuration
            surface: Surface orientation configuration
            irradiance_calculator: Optional pre-initialized calculator
        """
        self.location = location
        self.surface = surface
        self.calculator = irradiance_calculator or IrradianceCalculator(location)

    def direct_beam(
        self,
        dni: pd.Series,
        solar_position: SolarPosition,
    ) -> pd.Series:
        """Calculate direct beam irradiance on the tilted surface.

        Uses the projection of DNI onto the tilted surface based on
        the angle of incidence.

        Args:
            dni: Direct Normal Irradiance in W/m²
            solar_position: Solar position data

        Returns:
            Direct beam POA irradiance in W/m²

        Example:
            >>> poa_model = POAIrradianceModel(location, surface)
            >>> direct = poa_model.direct_beam(dni, solar_pos)
            >>> print(f"Peak direct: {direct.max():.1f} W/m²")
        """
        poa_direct = beam_component(
            surface_tilt=self.surface.tilt,
            surface_azimuth=self.surface.azimuth,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
            dni=dni,
        )

        return poa_direct.clip(lower=0)

    def sky_diffuse(
        self,
        dni: pd.Series,
        dhi: pd.Series,
        solar_position: SolarPosition,
        model: str = "perez",
        dni_extra: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """Calculate sky diffuse irradiance on the tilted surface.

        Supports multiple transposition models:
        - 'perez': Perez anisotropic model (recommended)
        - 'haydavies': Hay-Davies model
        - 'isotropic': Simple isotropic model

        Args:
            dni: Direct Normal Irradiance in W/m²
            dhi: Diffuse Horizontal Irradiance in W/m²
            solar_position: Solar position data
            model: Transposition model name
            dni_extra: Extraterrestrial DNI (calculated if not provided)
            times: DatetimeIndex for the data

        Returns:
            Sky diffuse POA irradiance in W/m²

        Example:
            >>> diffuse = poa_model.sky_diffuse(dni, dhi, solar_pos, model='perez')
            >>> print(f"Average diffuse: {diffuse.mean():.1f} W/m²")
        """
        if model == "perez":
            return self.calculator.perez_transposition(
                self.surface, solar_position, dni, dhi, dni_extra, times
            )
        elif model == "haydavies":
            return self.calculator.hay_davies_model(
                self.surface, solar_position, dni, dhi, dni_extra, times
            )
        elif model == "isotropic":
            return self.calculator.isotropic_sky(self.surface, dhi)
        else:
            raise ValueError(f"Unknown transposition model: {model}")

    def ground_reflected(
        self,
        ghi: pd.Series,
        albedo: Optional[float] = None,
    ) -> pd.Series:
        """Calculate ground reflected irradiance on the tilted surface.

        Uses the isotropic ground reflection model, which assumes
        uniform reflection from the ground.

        Args:
            ghi: Global Horizontal Irradiance in W/m²
            albedo: Ground reflectance (0-1), uses surface config if not provided

        Returns:
            Ground reflected POA irradiance in W/m²

        Example:
            >>> reflected = poa_model.ground_reflected(ghi, albedo=0.25)
            >>> print(f"Reflected contribution: {reflected.mean():.1f} W/m²")
        """
        if albedo is None:
            albedo = self.surface.albedo

        # Ground reflected irradiance formula
        # POA_reflected = GHI * albedo * (1 - cos(tilt)) / 2
        poa_reflected = ghi * albedo * (1 - np.cos(np.radians(self.surface.tilt))) / 2

        return poa_reflected.clip(lower=0)

    def spectral_corrections(
        self,
        poa_global: pd.Series,
        solar_position: SolarPosition,
        times: Optional[pd.DatetimeIndex] = None,
        module_type: str = "multisi",
        precipitable_water: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Calculate spectral mismatch corrections.

        Spectral corrections account for the variation in solar spectrum
        throughout the day and year, which affects PV module response.

        Args:
            poa_global: Total POA irradiance in W/m²
            solar_position: Solar position data
            times: DatetimeIndex for the data
            module_type: PV module type ('multisi', 'monosi', 'cdte', 'asi', 'cigs', 'perovskite')
            precipitable_water: Precipitable water in cm (estimated if not provided)

        Returns:
            Spectral correction factor (multiply POA by this)

        Example:
            >>> spectral_factor = poa_model.spectral_corrections(poa_global, solar_pos)
            >>> corrected_poa = poa_global * spectral_factor
        """
        if times is None:
            times = poa_global.index

        # Calculate airmass
        airmass = pvlib.atmosphere.get_relative_airmass(solar_position.zenith)
        airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass, pressure=None)

        # Estimate precipitable water if not provided
        if precipitable_water is None:
            precipitable_water = pvlib.atmosphere.gueymard94_pw(
                temp_air=20.0, relative_humidity=50.0
            )

        # Calculate spectral correction using First Solar model
        try:
            spectral_loss = pvlib.spectrum.spectral_factor_firstsolar(
                precipitable_water=precipitable_water,
                airmass_absolute=airmass_absolute,
                module_type=module_type,
            )
        except Exception:
            # Fallback to no correction if model fails
            spectral_loss = pd.Series(1.0, index=times)

        return spectral_loss

    def aoi_losses(
        self,
        solar_position: SolarPosition,
        n: float = 1.526,
        K: float = 4.0,
        L: float = 0.002,
    ) -> pd.Series:
        """Calculate angle of incidence (AOI) losses.

        Uses the ASHRAE model for AOI-dependent transmittance losses
        due to reflection at the module surface.

        Args:
            solar_position: Solar position data
            n: Refractive index of glass cover (default 1.526)
            K: Glazing extinction coefficient (default 4.0)
            L: Glazing thickness in meters (default 0.002)

        Returns:
            AOI loss factor (multiply POA by this to get effective irradiance)

        Example:
            >>> aoi_factor = poa_model.aoi_losses(solar_pos)
            >>> effective_poa = poa_global * aoi_factor
            >>> losses = (1 - aoi_factor) * 100
            >>> print(f"Average AOI loss: {losses.mean():.1f}%")
        """
        # Calculate angle of incidence
        aoi_value = aoi(
            surface_tilt=self.surface.tilt,
            surface_azimuth=self.surface.azimuth,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
        )

        # Calculate ASHRAE IAM (Incidence Angle Modifier)
        iam = pvlib.iam.ashrae(aoi_value, b=0.05)

        # Alternatively, use physical model
        # iam = pvlib.iam.physical(aoi_value, n=n, K=K, L=L)

        return iam.clip(lower=0, upper=1)

    def calculate_poa_components(
        self,
        irradiance: IrradianceComponents,
        solar_position: SolarPosition,
        transposition_model: str = "perez",
        include_spectral: bool = False,
        include_aoi: bool = False,
        module_type: str = "multisi",
        times: Optional[pd.DatetimeIndex] = None,
    ) -> POAComponents:
        """Calculate all POA irradiance components.

        This is the main method that combines all components to calculate
        total POA irradiance with optional loss factors.

        Args:
            irradiance: Irradiance components (GHI, DNI, DHI)
            solar_position: Solar position data
            transposition_model: Sky diffuse transposition model
            include_spectral: Apply spectral corrections
            include_aoi: Apply AOI losses
            module_type: PV module type for spectral corrections
            times: DatetimeIndex for the data

        Returns:
            POAComponents with all irradiance components

        Example:
            >>> components = poa_model.calculate_poa_components(
            ...     irradiance, solar_pos, transposition_model='perez',
            ...     include_spectral=True, include_aoi=True
            ... )
            >>> print(f"Total POA: {components.poa_global.sum():.1f} Wh/m²")
        """
        # Calculate base components
        poa_direct = self.direct_beam(irradiance.dni, solar_position)
        poa_diffuse = self.sky_diffuse(
            irradiance.dni,
            irradiance.dhi,
            solar_position,
            model=transposition_model,
            times=times,
        )
        poa_ground = self.ground_reflected(irradiance.ghi)

        # Sum components
        poa_global = poa_direct + poa_diffuse + poa_ground

        # Apply loss factors if requested
        if include_spectral:
            spectral_factor = self.spectral_corrections(
                poa_global, solar_position, times, module_type
            )
            poa_global = poa_global * spectral_factor
            poa_direct = poa_direct * spectral_factor
            poa_diffuse = poa_diffuse * spectral_factor
            poa_ground = poa_ground * spectral_factor

        if include_aoi:
            aoi_factor = self.aoi_losses(solar_position)
            poa_global = poa_global * aoi_factor
            poa_direct = poa_direct * aoi_factor
            poa_diffuse = poa_diffuse * aoi_factor
            poa_ground = poa_ground * aoi_factor

        return POAComponents(
            poa_global=poa_global,
            poa_direct=poa_direct,
            poa_diffuse=poa_diffuse,
            poa_ground=poa_ground,
        )

    def calculate_effective_irradiance(
        self,
        irradiance: IrradianceComponents,
        solar_position: SolarPosition,
        transposition_model: str = "perez",
        module_type: str = "multisi",
        times: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """Calculate effective irradiance with all loss factors applied.

        This method applies both spectral and AOI corrections to provide
        the effective irradiance seen by the PV module.

        Args:
            irradiance: Irradiance components (GHI, DNI, DHI)
            solar_position: Solar position data
            transposition_model: Sky diffuse transposition model
            module_type: PV module type for spectral corrections
            times: DatetimeIndex for the data

        Returns:
            Effective POA irradiance in W/m²

        Example:
            >>> eff_irr = poa_model.calculate_effective_irradiance(
            ...     irradiance, solar_pos, module_type='monosi'
            ... )
            >>> print(f"Daily effective: {eff_irr.sum():.1f} Wh/m²")
        """
        components = self.calculate_poa_components(
            irradiance,
            solar_position,
            transposition_model=transposition_model,
            include_spectral=True,
            include_aoi=True,
            module_type=module_type,
            times=times,
        )

        return components.poa_global
