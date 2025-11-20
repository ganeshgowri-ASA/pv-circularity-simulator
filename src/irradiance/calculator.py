"""Irradiance calculation and decomposition models."""

from typing import Optional, Union

import numpy as np
import pandas as pd
import pvlib
from pvlib.irradiance import (
    aoi,
    beam_component,
    disc,
    dirint,
    erbs,
    get_extra_radiation,
    haydavies,
    isotropic,
    perez,
)

from .models import IrradianceComponents, LocationConfig, SolarPosition, SurfaceConfig


class IrradianceCalculator:
    """Calculator for solar irradiance decomposition and transposition.

    This class provides methods for:
    - Decomposing GHI into DNI and DHI components
    - Transposing irradiance to tilted surfaces using various models
    - Calculating anisotropic corrections

    The calculator uses pvlib-python for robust, well-tested algorithms.
    """

    def __init__(self, location: LocationConfig):
        """Initialize the irradiance calculator.

        Args:
            location: Geographic location configuration
        """
        self.location = location
        self.pvlib_location = pvlib.location.Location(
            latitude=location.latitude,
            longitude=location.longitude,
            altitude=location.altitude,
            tz=location.timezone,
            name=location.name,
        )

    def get_solar_position(
        self, times: pd.DatetimeIndex, method: str = "nrel_numpy"
    ) -> SolarPosition:
        """Calculate solar position for given times.

        Args:
            times: DatetimeIndex with timezone information
            method: Solar position algorithm ('nrel_numpy', 'ephemeris', 'pyephem')

        Returns:
            SolarPosition object with zenith, azimuth, and elevation angles

        Example:
            >>> times = pd.date_range('2024-01-01', periods=24, freq='h', tz='UTC')
            >>> calc = IrradianceCalculator(location)
            >>> solar_pos = calc.get_solar_position(times)
            >>> print(solar_pos.zenith)
        """
        solpos = self.pvlib_location.get_solarposition(times, method=method)

        return SolarPosition(
            zenith=solpos["zenith"],
            azimuth=solpos["azimuth"],
            elevation=solpos["elevation"],
            equation_of_time=solpos.get("equation_of_time"),
        )

    def ghi_dni_dhi_decomposition(
        self,
        ghi: pd.Series,
        solar_position: Optional[SolarPosition] = None,
        times: Optional[pd.DatetimeIndex] = None,
        model: str = "dirint",
    ) -> IrradianceComponents:
        """Decompose Global Horizontal Irradiance into DNI and DHI components.

        Uses empirical models to estimate direct and diffuse components from GHI.
        Supported models: 'dirint', 'disc', 'erbs'

        Args:
            ghi: Global Horizontal Irradiance time series in W/m²
            solar_position: Pre-calculated solar position (optional)
            times: DatetimeIndex for GHI data (required if solar_position not provided)
            model: Decomposition model ('dirint', 'disc', 'erbs')

        Returns:
            IrradianceComponents with GHI, DNI, and DHI

        Raises:
            ValueError: If neither solar_position nor times is provided

        Example:
            >>> ghi_data = pd.Series([0, 200, 600, 800, 600, 200, 0])
            >>> components = calc.ghi_dni_dhi_decomposition(ghi_data, times=times)
            >>> print(f"DNI: {components.dni.mean():.1f} W/m²")
        """
        # Get solar position if not provided
        if solar_position is None:
            if times is None:
                raise ValueError("Either solar_position or times must be provided")
            solar_position = self.get_solar_position(times)

        zenith = solar_position.zenith

        # Calculate extraterrestrial radiation
        if times is not None:
            dni_extra = get_extra_radiation(times)
        else:
            dni_extra = get_extra_radiation(ghi.index)

        # Apply decomposition model
        if model == "dirint":
            # DIRINT model (recommended for hourly data)
            dni = dirint(ghi, zenith, times if times is not None else ghi.index)
        elif model == "disc":
            # DISC model (simpler, faster)
            dni = disc(ghi, zenith, times if times is not None else ghi.index)["dni"]
        elif model == "erbs":
            # Erbs model (good for daily or hourly data)
            result = erbs(ghi, zenith, times if times is not None else ghi.index)
            dni = result["dni"]
        else:
            raise ValueError(f"Unknown decomposition model: {model}")

        # Calculate DHI from GHI and DNI
        # GHI = DNI * cos(zenith) + DHI
        dhi = ghi - dni * np.cos(np.radians(zenith))
        dhi = dhi.clip(lower=0)  # Ensure non-negative

        return IrradianceComponents(ghi=ghi, dni=dni, dhi=dhi)

    def perez_transposition(
        self,
        surface: SurfaceConfig,
        solar_position: SolarPosition,
        dni: pd.Series,
        dhi: pd.Series,
        dni_extra: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """Calculate sky diffuse irradiance using the Perez anisotropic model.

        The Perez model is one of the most accurate transposition models,
        accounting for circumsolar and horizon brightening effects.

        Args:
            surface: Surface orientation configuration
            solar_position: Solar position data
            dni: Direct Normal Irradiance in W/m²
            dhi: Diffuse Horizontal Irradiance in W/m²
            dni_extra: Extraterrestrial DNI (calculated if not provided)
            times: DatetimeIndex for the data

        Returns:
            Sky diffuse irradiance on tilted surface in W/m²

        Example:
            >>> sky_diff = calc.perez_transposition(surface, solar_pos, dni, dhi)
            >>> print(f"Sky diffuse: {sky_diff.mean():.1f} W/m²")
        """
        # Calculate extraterrestrial radiation if not provided
        if dni_extra is None:
            if times is None:
                times = dni.index
            dni_extra = get_extra_radiation(times)

        # Calculate angle of incidence
        aoi_value = aoi(
            surface_tilt=surface.tilt,
            surface_azimuth=surface.azimuth,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
        )

        # Apply Perez model
        sky_diffuse = perez(
            surface_tilt=surface.tilt,
            surface_azimuth=surface.azimuth,
            dhi=dhi,
            dni=dni,
            dni_extra=dni_extra,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
            airmass=pvlib.atmosphere.get_relative_airmass(solar_position.zenith),
        )

        return sky_diffuse

    def hay_davies_model(
        self,
        surface: SurfaceConfig,
        solar_position: SolarPosition,
        dni: pd.Series,
        dhi: pd.Series,
        dni_extra: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """Calculate sky diffuse irradiance using the Hay-Davies model.

        The Hay-Davies model separates diffuse irradiance into isotropic
        and circumsolar components.

        Args:
            surface: Surface orientation configuration
            solar_position: Solar position data
            dni: Direct Normal Irradiance in W/m²
            dhi: Diffuse Horizontal Irradiance in W/m²
            dni_extra: Extraterrestrial DNI (calculated if not provided)
            times: DatetimeIndex for the data

        Returns:
            Sky diffuse irradiance on tilted surface in W/m²

        Example:
            >>> sky_diff = calc.hay_davies_model(surface, solar_pos, dni, dhi)
            >>> print(f"Hay-Davies diffuse: {sky_diff.mean():.1f} W/m²")
        """
        # Calculate extraterrestrial radiation if not provided
        if dni_extra is None:
            if times is None:
                times = dni.index
            dni_extra = get_extra_radiation(times)

        # Apply Hay-Davies model
        sky_diffuse = haydavies(
            surface_tilt=surface.tilt,
            surface_azimuth=surface.azimuth,
            dhi=dhi,
            dni=dni,
            dni_extra=dni_extra,
            solar_zenith=solar_position.zenith,
            solar_azimuth=solar_position.azimuth,
        )

        return sky_diffuse

    def isotropic_sky(
        self, surface: SurfaceConfig, dhi: pd.Series
    ) -> pd.Series:
        """Calculate sky diffuse irradiance using the isotropic model.

        The isotropic model assumes uniform diffuse irradiance from the sky dome.
        This is the simplest transposition model.

        Args:
            surface: Surface orientation configuration
            dhi: Diffuse Horizontal Irradiance in W/m²

        Returns:
            Sky diffuse irradiance on tilted surface in W/m²

        Example:
            >>> sky_diff = calc.isotropic_sky(surface, dhi)
            >>> print(f"Isotropic diffuse: {sky_diff.mean():.1f} W/m²")
        """
        return isotropic(surface.tilt, dhi)

    def anisotropic_corrections(
        self,
        surface: SurfaceConfig,
        solar_position: SolarPosition,
        dni: pd.Series,
        dhi: pd.Series,
        model: str = "perez",
        dni_extra: Optional[pd.Series] = None,
        times: Optional[pd.DatetimeIndex] = None,
    ) -> dict[str, pd.Series]:
        """Calculate anisotropic corrections for sky diffuse irradiance.

        Compares isotropic model with anisotropic models to show the
        impact of circumsolar and horizon brightening effects.

        Args:
            surface: Surface orientation configuration
            solar_position: Solar position data
            dni: Direct Normal Irradiance in W/m²
            dhi: Diffuse Horizontal Irradiance in W/m²
            model: Anisotropic model ('perez' or 'haydavies')
            dni_extra: Extraterrestrial DNI (calculated if not provided)
            times: DatetimeIndex for the data

        Returns:
            Dictionary with 'isotropic', 'anisotropic', 'correction_factor',
            and 'absolute_correction' Series

        Example:
            >>> corrections = calc.anisotropic_corrections(surface, solar_pos, dni, dhi)
            >>> avg_correction = corrections['correction_factor'].mean()
            >>> print(f"Average correction factor: {avg_correction:.3f}")
        """
        # Calculate isotropic component
        iso_diffuse = self.isotropic_sky(surface, dhi)

        # Calculate anisotropic component
        if model == "perez":
            aniso_diffuse = self.perez_transposition(
                surface, solar_position, dni, dhi, dni_extra, times
            )
        elif model == "haydavies":
            aniso_diffuse = self.hay_davies_model(
                surface, solar_position, dni, dhi, dni_extra, times
            )
        else:
            raise ValueError(f"Unknown anisotropic model: {model}")

        # Calculate corrections
        correction_factor = aniso_diffuse / iso_diffuse.replace(0, np.nan)
        absolute_correction = aniso_diffuse - iso_diffuse

        return {
            "isotropic": iso_diffuse,
            "anisotropic": aniso_diffuse,
            "correction_factor": correction_factor,
            "absolute_correction": absolute_correction,
        }

    def calculate_clearness_index(
        self, ghi: pd.Series, times: Optional[pd.DatetimeIndex] = None
    ) -> pd.Series:
        """Calculate clearness index (kt) - ratio of GHI to extraterrestrial irradiance.

        The clearness index indicates atmospheric transparency and cloud cover.
        Values range from 0 (complete overcast) to ~0.8 (clear sky).

        Args:
            ghi: Global Horizontal Irradiance in W/m²
            times: DatetimeIndex (uses ghi.index if not provided)

        Returns:
            Clearness index (dimensionless)

        Example:
            >>> kt = calc.calculate_clearness_index(ghi)
            >>> print(f"Average clearness: {kt.mean():.3f}")
        """
        if times is None:
            times = ghi.index

        dni_extra = get_extra_radiation(times)
        solar_pos = self.get_solar_position(times)

        # Extraterrestrial horizontal irradiance
        ghi_extra = dni_extra * np.cos(np.radians(solar_pos.zenith))
        ghi_extra = ghi_extra.clip(lower=1)  # Avoid division by zero

        kt = ghi / ghi_extra
        return kt.clip(lower=0, upper=1)
