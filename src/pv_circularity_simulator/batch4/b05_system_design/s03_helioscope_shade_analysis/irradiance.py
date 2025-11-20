"""
Irradiance on Surface calculations with comprehensive transposition models.

This module provides plane-of-array (POA) irradiance calculations using
various transposition models including Perez, Hay-Davies, and isotropic models.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .models import (
    ArrayGeometry,
    IrradianceComponents,
    Location,
    SunPosition,
    TranspositionModel,
    AOIModel,
)

logger = logging.getLogger(__name__)


class IrradianceOnSurface:
    """
    Calculate plane-of-array irradiance using various transposition models.

    This class converts global horizontal irradiance (GHI) to plane-of-array
    (POA) irradiance considering direct, diffuse, and ground-reflected components.
    """

    def __init__(
        self,
        location: Location,
        array_geometry: ArrayGeometry,
        transposition_model: TranspositionModel = TranspositionModel.PEREZ,
        aoi_model: AOIModel = AOIModel.ASHRAE
    ):
        """
        Initialize the irradiance calculator.

        Args:
            location: Geographic location
            array_geometry: PV array geometry configuration
            transposition_model: Model for diffuse transposition
            aoi_model: Model for angle-of-incidence corrections
        """
        self.location = location
        self.array_geometry = array_geometry
        self.transposition_model = transposition_model
        self.aoi_model = aoi_model

        logger.info(
            f"Initialized IrradianceOnSurface with {transposition_model.value} "
            f"transposition and {aoi_model.value} AOI model"
        )

    def poa_irradiance(
        self,
        irradiance: IrradianceComponents,
        sun_position: SunPosition,
        surface_tilt: Optional[float] = None,
        surface_azimuth: Optional[float] = None,
        albedo: float = 0.2
    ) -> IrradianceComponents:
        """
        Calculate plane-of-array irradiance components.

        Computes direct, diffuse, and ground-reflected POA irradiance using
        the configured transposition model.

        Args:
            irradiance: GHI, DNI, DHI components
            sun_position: Solar position data
            surface_tilt: Surface tilt angle (degrees). If None, uses array_geometry
            surface_azimuth: Surface azimuth (degrees). If None, uses array_geometry
            albedo: Ground reflectance (0-1)

        Returns:
            IrradianceComponents with POA values calculated
        """
        # Use array geometry if surface angles not specified
        tilt = surface_tilt if surface_tilt is not None else self.array_geometry.tilt
        azimuth = surface_azimuth if surface_azimuth is not None else self.array_geometry.azimuth

        # Calculate angle of incidence
        aoi = self._calculate_aoi(sun_position, tilt, azimuth)

        # Calculate direct POA irradiance
        poa_direct = self._calculate_poa_direct(irradiance.dni, sun_position, aoi)

        # Calculate diffuse POA irradiance using selected model
        poa_diffuse = self._calculate_poa_diffuse(
            irradiance,
            sun_position,
            tilt,
            aoi
        )

        # Calculate ground-reflected POA irradiance
        poa_ground = self._calculate_poa_ground_reflected(
            irradiance.ghi,
            tilt,
            albedo
        )

        # Total POA irradiance
        poa_global = poa_direct + poa_diffuse + poa_ground

        # Calculate AOI modifier
        aoi_modifier = self._calculate_aoi_modifier(aoi)

        # Apply AOI correction
        poa_global_corrected = poa_global * aoi_modifier

        # Update irradiance components
        irradiance.poa_direct = poa_direct
        irradiance.poa_diffuse = poa_diffuse
        irradiance.poa_ground = poa_ground
        irradiance.poa_global = poa_global_corrected
        irradiance.aoi = aoi
        irradiance.aoi_modifier = aoi_modifier

        return irradiance

    def transposition_model(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        sun_position: SunPosition,
        surface_tilt: float,
        surface_azimuth: float,
        model: Optional[TranspositionModel] = None
    ) -> Tuple[float, float, float]:
        """
        Apply transposition model to convert GHI to POA.

        Args:
            ghi: Global horizontal irradiance (W/m²)
            dni: Direct normal irradiance (W/m²)
            dhi: Diffuse horizontal irradiance (W/m²)
            sun_position: Solar position
            surface_tilt: Surface tilt angle (degrees)
            surface_azimuth: Surface azimuth (degrees)
            model: Transposition model to use (if None, uses instance default)

        Returns:
            Tuple of (poa_direct, poa_diffuse, poa_ground) in W/m²
        """
        model = model or self.transposition_model

        # Calculate AOI
        aoi = self._calculate_aoi(sun_position, surface_tilt, surface_azimuth)

        # Direct component
        poa_direct = self._calculate_poa_direct(dni, sun_position, aoi)

        # Diffuse component (model-dependent)
        if model == TranspositionModel.ISOTROPIC:
            poa_diffuse = self._isotropic_diffuse(dhi, surface_tilt)
        elif model == TranspositionModel.PEREZ:
            poa_diffuse = self._perez_diffuse(
                ghi, dni, dhi, sun_position, surface_tilt, surface_azimuth, aoi
            )
        elif model == TranspositionModel.HAY_DAVIES:
            poa_diffuse = self._hay_davies_diffuse(
                dni, dhi, sun_position, surface_tilt, aoi
            )
        elif model == TranspositionModel.REINDL:
            poa_diffuse = self._reindl_diffuse(
                ghi, dni, dhi, sun_position, surface_tilt, aoi
            )
        elif model == TranspositionModel.KLUCHER:
            poa_diffuse = self._klucher_diffuse(
                ghi, dhi, sun_position, surface_tilt, surface_azimuth, aoi
            )
        else:
            poa_diffuse = self._isotropic_diffuse(dhi, surface_tilt)

        # Ground-reflected component
        poa_ground = self._calculate_poa_ground_reflected(
            ghi, surface_tilt, self.location.albedo if hasattr(self.location, 'albedo') else 0.2
        )

        return poa_direct, poa_diffuse, poa_ground

    def aoi_correction(self, aoi: float, model: Optional[AOIModel] = None) -> float:
        """
        Calculate angle-of-incidence correction factor.

        Args:
            aoi: Angle of incidence in degrees
            model: AOI model to use (if None, uses instance default)

        Returns:
            AOI correction factor (0-1)
        """
        model = model or self.aoi_model
        return self._calculate_aoi_modifier(aoi, model)

    def soiling_model(
        self,
        base_soiling_rate: float = 0.02,
        days_since_cleaning: int = 30,
        seasonal_factor: float = 1.0
    ) -> float:
        """
        Calculate soiling loss factor.

        Args:
            base_soiling_rate: Base daily soiling rate (fraction/day)
            days_since_cleaning: Days since last cleaning
            seasonal_factor: Seasonal multiplier (1.0 = average)

        Returns:
            Soiling loss factor (0-1, where 1 = no loss)
        """
        # Linear soiling accumulation model
        soiling_loss = base_soiling_rate * days_since_cleaning * seasonal_factor

        # Clamp to reasonable range
        soiling_loss = min(soiling_loss, 0.3)  # Max 30% loss

        return 1.0 - soiling_loss

    def spectral_correction(
        self,
        air_mass: float,
        module_type: str = "crystalline_silicon"
    ) -> float:
        """
        Calculate spectral mismatch correction.

        Args:
            air_mass: Relative air mass
            module_type: Type of PV module (crystalline_silicon, thin_film_cdte, etc.)

        Returns:
            Spectral correction factor (typically 0.95-1.05)
        """
        # Simplified spectral correction based on air mass
        # More sophisticated models would use detailed spectral data

        if module_type == "crystalline_silicon":
            # c-Si benefits slightly at higher air mass
            correction = 1.0 + 0.001 * (air_mass - 1.5)
        elif module_type == "thin_film_cdte":
            # CdTe less sensitive to spectral changes
            correction = 1.0 + 0.0005 * (air_mass - 1.5)
        elif module_type == "thin_film_cigs":
            # CIGS moderate sensitivity
            correction = 1.0 + 0.0008 * (air_mass - 1.5)
        else:
            # Default to c-Si behavior
            correction = 1.0 + 0.001 * (air_mass - 1.5)

        # Clamp to reasonable range
        correction = np.clip(correction, 0.95, 1.05)

        return correction

    # Private calculation methods

    def _calculate_aoi(
        self,
        sun_position: SunPosition,
        surface_tilt: float,
        surface_azimuth: float
    ) -> float:
        """
        Calculate angle of incidence between sun and surface normal.

        Args:
            sun_position: Solar position
            surface_tilt: Surface tilt from horizontal (degrees)
            surface_azimuth: Surface azimuth (degrees, 0=North)

        Returns:
            Angle of incidence in degrees
        """
        # Convert to radians
        sun_zenith_rad = np.radians(sun_position.zenith)
        sun_azimuth_rad = np.radians(sun_position.azimuth)
        surface_tilt_rad = np.radians(surface_tilt)
        surface_azimuth_rad = np.radians(surface_azimuth)

        # Calculate AOI using spherical geometry
        cos_aoi = (
            np.cos(sun_zenith_rad) * np.cos(surface_tilt_rad) +
            np.sin(sun_zenith_rad) * np.sin(surface_tilt_rad) *
            np.cos(sun_azimuth_rad - surface_azimuth_rad)
        )

        # Clamp to valid range to avoid numerical errors
        cos_aoi = np.clip(cos_aoi, -1.0, 1.0)

        aoi_rad = np.arccos(cos_aoi)
        aoi = np.degrees(aoi_rad)

        return aoi

    def _calculate_poa_direct(
        self,
        dni: float,
        sun_position: SunPosition,
        aoi: float
    ) -> float:
        """Calculate direct POA irradiance."""
        if sun_position.elevation <= 0 or aoi >= 90:
            return 0.0

        poa_direct = dni * np.cos(np.radians(aoi))
        return max(0.0, poa_direct)

    def _calculate_poa_diffuse(
        self,
        irradiance: IrradianceComponents,
        sun_position: SunPosition,
        surface_tilt: float,
        aoi: float
    ) -> float:
        """Calculate diffuse POA irradiance using configured model."""
        if self.transposition_model == TranspositionModel.PEREZ:
            return self._perez_diffuse(
                irradiance.ghi,
                irradiance.dni,
                irradiance.dhi,
                sun_position,
                surface_tilt,
                self.array_geometry.azimuth,
                aoi
            )
        elif self.transposition_model == TranspositionModel.HAY_DAVIES:
            return self._hay_davies_diffuse(
                irradiance.dni,
                irradiance.dhi,
                sun_position,
                surface_tilt,
                aoi
            )
        else:
            return self._isotropic_diffuse(irradiance.dhi, surface_tilt)

    def _isotropic_diffuse(self, dhi: float, surface_tilt: float) -> float:
        """Isotropic sky diffuse model."""
        return dhi * (1 + np.cos(np.radians(surface_tilt))) / 2

    def _perez_diffuse(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        sun_position: SunPosition,
        surface_tilt: float,
        surface_azimuth: float,
        aoi: float
    ) -> float:
        """
        Perez diffuse irradiance model.

        This is the most accurate model for diffuse irradiance transposition,
        accounting for circumsolar and horizon brightening effects.
        """
        if dhi <= 0 or sun_position.elevation <= 0:
            return 0.0

        # Calculate extraterrestrial irradiance
        dni_extra = 1367.0  # Solar constant W/m²

        # Sky clearness parameter (epsilon)
        # Add small value to avoid division by zero
        epsilon = ((dhi + dni) / (dhi + 1e-6) + 5.535e-6 * sun_position.zenith**3) / (1 + 5.535e-6 * sun_position.zenith**3)

        # Sky brightness parameter (delta)
        air_mass = self._calculate_air_mass(sun_position.zenith)
        delta = dhi * air_mass / dni_extra

        # Perez coefficients (based on sky clearness bins)
        f11, f12, f13, f21, f22, f23 = self._get_perez_coefficients(epsilon)

        # Circumsolar component
        a = max(0.0, np.cos(np.radians(aoi)))
        b = max(0.087, np.cos(np.radians(sun_position.zenith)))
        f1 = max(0.0, f11 + f12 * delta + f13 * np.radians(sun_position.zenith))

        # Horizon brightness component
        surface_tilt_rad = np.radians(surface_tilt)
        f2 = f21 + f22 * delta + f23 * np.radians(sun_position.zenith)

        # Diffuse irradiance on tilted surface
        poa_diffuse = dhi * (
            (1 - f1) * (1 + np.cos(surface_tilt_rad)) / 2 +
            f1 * a / b +
            f2 * np.sin(surface_tilt_rad)
        )

        return max(0.0, poa_diffuse)

    def _hay_davies_diffuse(
        self,
        dni: float,
        dhi: float,
        sun_position: SunPosition,
        surface_tilt: float,
        aoi: float
    ) -> float:
        """Hay-Davies diffuse irradiance model."""
        if dhi <= 0 or sun_position.elevation <= 0:
            return 0.0

        # Anisotropy index
        dni_extra = 1367.0
        ai = dni / dni_extra

        # Circumsolar diffuse
        rb = max(0.0, np.cos(np.radians(aoi))) / max(0.087, np.cos(np.radians(sun_position.zenith)))
        circumsolar = dhi * ai * rb

        # Isotropic diffuse
        surface_tilt_rad = np.radians(surface_tilt)
        isotropic = dhi * (1 - ai) * (1 + np.cos(surface_tilt_rad)) / 2

        poa_diffuse = circumsolar + isotropic

        return max(0.0, poa_diffuse)

    def _reindl_diffuse(
        self,
        ghi: float,
        dni: float,
        dhi: float,
        sun_position: SunPosition,
        surface_tilt: float,
        aoi: float
    ) -> float:
        """Reindl diffuse irradiance model."""
        if dhi <= 0 or sun_position.elevation <= 0:
            return 0.0

        # Start with Hay-Davies
        poa_diffuse = self._hay_davies_diffuse(dni, dhi, sun_position, surface_tilt, aoi)

        # Add horizon brightening term
        surface_tilt_rad = np.radians(surface_tilt)
        sun_zenith_rad = np.radians(sun_position.zenith)

        horizon_term = dhi * np.sqrt(dni / (dni + dhi + 1e-6)) * np.sin(surface_tilt_rad / 2)**3

        poa_diffuse += horizon_term

        return max(0.0, poa_diffuse)

    def _klucher_diffuse(
        self,
        ghi: float,
        dhi: float,
        sun_position: SunPosition,
        surface_tilt: float,
        surface_azimuth: float,
        aoi: float
    ) -> float:
        """Klucher diffuse irradiance model."""
        if dhi <= 0:
            return 0.0

        surface_tilt_rad = np.radians(surface_tilt)

        # Isotropic term
        isotropic = dhi * (1 + np.cos(surface_tilt_rad)) / 2

        if sun_position.elevation <= 0:
            return isotropic

        # Modulating function
        f = 1 - (dhi / (ghi + 1e-6))**2

        # Angular term
        sun_azimuth_rad = np.radians(sun_position.azimuth)
        surface_azimuth_rad = np.radians(surface_azimuth)

        angular_term = 1 + f * np.sin(surface_tilt_rad / 2)**3
        angular_term *= 1 + f * np.cos(np.radians(aoi))**2 * np.sin(np.radians(sun_position.zenith))**3

        poa_diffuse = isotropic * angular_term

        return max(0.0, poa_diffuse)

    def _calculate_poa_ground_reflected(
        self,
        ghi: float,
        surface_tilt: float,
        albedo: float
    ) -> float:
        """Calculate ground-reflected POA irradiance."""
        surface_tilt_rad = np.radians(surface_tilt)
        poa_ground = ghi * albedo * (1 - np.cos(surface_tilt_rad)) / 2

        return max(0.0, poa_ground)

    def _calculate_aoi_modifier(
        self,
        aoi: float,
        model: Optional[AOIModel] = None
    ) -> float:
        """Calculate AOI correction modifier."""
        model = model or self.aoi_model

        if aoi >= 90:
            return 0.0

        if model == AOIModel.ASHRAE:
            return self._ashrae_aoi_modifier(aoi)
        elif model == AOIModel.PHYSICAL:
            return self._physical_aoi_modifier(aoi)
        elif model == AOIModel.SANDIA:
            return self._sandia_aoi_modifier(aoi)
        elif model == AOIModel.MARTIN_RUIZ:
            return self._martin_ruiz_aoi_modifier(aoi)
        else:
            return self._ashrae_aoi_modifier(aoi)

    def _ashrae_aoi_modifier(self, aoi: float) -> float:
        """ASHRAE AOI modifier (simple cosine with coefficient)."""
        b0 = 0.05  # Typical value for glass-covered modules
        modifier = 1 - b0 * (1 / np.cos(np.radians(aoi)) - 1)

        return max(0.0, min(1.0, modifier))

    def _physical_aoi_modifier(self, aoi: float) -> float:
        """Physical (Fresnel) AOI modifier."""
        # Simplified Fresnel reflection for air-glass interface
        n = 1.526  # Refractive index of glass
        aoi_rad = np.radians(aoi)

        # Snell's law
        theta_r_rad = np.arcsin(np.sin(aoi_rad) / n)

        # Fresnel equations
        rs = np.sin(theta_r_rad - aoi_rad)**2 / np.sin(theta_r_rad + aoi_rad)**2
        rp = np.tan(theta_r_rad - aoi_rad)**2 / np.tan(theta_r_rad + aoi_rad)**2

        reflectance = (rs + rp) / 2
        transmittance = 1 - reflectance

        # Account for absorption in glass
        absorption = 0.04  # Typical value
        modifier = transmittance * (1 - absorption)

        return max(0.0, min(1.0, modifier))

    def _sandia_aoi_modifier(self, aoi: float) -> float:
        """Sandia AOI modifier."""
        # Typical Sandia coefficients for glass-covered modules
        b0 = 0.05
        b1 = 0.04
        b2 = -0.002
        b3 = 0.0002
        b4 = -0.00001
        b5 = 0.0000004

        aoi_rad = np.radians(aoi)

        modifier = 1 + b0 * (1 / np.cos(aoi_rad) - 1)
        modifier += b1 * (1 / np.cos(aoi_rad) - 1)**2
        modifier += b2 * (1 / np.cos(aoi_rad) - 1)**3
        modifier += b3 * (1 / np.cos(aoi_rad) - 1)**4
        modifier += b4 * (1 / np.cos(aoi_rad) - 1)**5
        modifier += b5 * (1 / np.cos(aoi_rad) - 1)**6

        modifier = 1 / modifier

        return max(0.0, min(1.0, modifier))

    def _martin_ruiz_aoi_modifier(self, aoi: float) -> float:
        """Martin-Ruiz AOI modifier."""
        # Typical parameter value
        a_r = 0.16

        aoi_rad = np.radians(aoi)

        modifier = (1 - np.exp(-np.cos(aoi_rad) / a_r)) / (1 - np.exp(-1 / a_r))

        return max(0.0, min(1.0, modifier))

    def _get_perez_coefficients(self, epsilon: float) -> Tuple[float, float, float, float, float, float]:
        """
        Get Perez model coefficients based on sky clearness.

        Returns f11, f12, f13, f21, f22, f23
        """
        # Perez coefficient table
        coefficients = [
            #  f11    f12    f13    f21    f22    f23
            [-0.008, 0.588, -0.062, -0.060, 0.072, -0.022],  # epsilon < 1.065
            [0.130, 0.683, -0.151, -0.019, 0.066, -0.029],   # 1.065 <= epsilon < 1.230
            [0.330, 0.487, -0.221, 0.055, -0.064, -0.026],   # 1.230 <= epsilon < 1.500
            [0.568, 0.187, -0.295, 0.109, -0.152, -0.014],   # 1.500 <= epsilon < 1.950
            [0.873, -0.392, -0.362, 0.226, -0.462, 0.001],   # 1.950 <= epsilon < 2.800
            [1.132, -1.237, -0.412, 0.288, -0.823, 0.056],   # 2.800 <= epsilon < 4.500
            [1.060, -1.600, -0.359, 0.264, -1.127, 0.131],   # 4.500 <= epsilon < 6.200
            [0.678, -0.327, -0.250, 0.156, -1.377, 0.251],   # epsilon >= 6.200
        ]

        # Select coefficient set based on epsilon
        if epsilon < 1.065:
            idx = 0
        elif epsilon < 1.230:
            idx = 1
        elif epsilon < 1.500:
            idx = 2
        elif epsilon < 1.950:
            idx = 3
        elif epsilon < 2.800:
            idx = 4
        elif epsilon < 4.500:
            idx = 5
        elif epsilon < 6.200:
            idx = 6
        else:
            idx = 7

        return tuple(coefficients[idx])

    def _calculate_air_mass(self, zenith: float) -> float:
        """
        Calculate relative air mass.

        Uses Kasten-Young formula for air mass calculation.
        """
        if zenith >= 90:
            return 38.0  # Maximum reasonable air mass

        zenith_rad = np.radians(zenith)

        # Kasten-Young formula
        am = 1 / (np.cos(zenith_rad) + 0.50572 * (96.07995 - zenith)**(-1.6364))

        return min(am, 38.0)
