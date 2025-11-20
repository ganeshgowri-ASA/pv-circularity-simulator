"""Fire safety testing and classification per IEC 61730-2 and UL 790.

This module implements fire safety tests for PV modules, including spread of flame,
fire penetration, and flying brand tests. Fire classifications (Class A, B, C) are
determined based on test results per UL 790 and IEC 61730-2 Annex C.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from loguru import logger

from ..models.safety_models import (
    SpreadOfFlameTestResult,
    FirePenetrationTestResult,
    FireBrandTestResult,
    FireSafetyTestResult,
    FireClass,
    TestStatus,
)


class FireSafetyClassification:
    """Implements fire safety testing and classification per IEC 61730-2 / UL 790.

    This class provides methods for executing fire safety tests and determining
    fire classification (Class A, B, C, or Not Rated) based on test results.
    Testing follows UL 790 procedures as referenced in IEC 61730-2 Annex C.

    Attributes:
        module_id: Unique identifier for the module under test.
        module_area_m2: Module area in square meters.
        roof_mounting: Whether module is intended for roof mounting.
        backsheet_material: Type of backsheet material (affects flammability).
        frame_material: Type of frame material.
    """

    def __init__(
        self,
        module_id: str,
        module_area_m2: float,
        roof_mounting: bool = True,
        backsheet_material: str = "PET/EVA/PET",
        frame_material: str = "Aluminum",
    ) -> None:
        """Initialize fire safety tester.

        Args:
            module_id: Unique module identifier.
            module_area_m2: Module area in square meters.
            roof_mounting: Whether module is for roof mounting (default: True).
            backsheet_material: Backsheet material type (default: "PET/EVA/PET").
            frame_material: Frame material type (default: "Aluminum").
        """
        self.module_id = module_id
        self.module_area_m2 = module_area_m2
        self.roof_mounting = roof_mounting
        self.backsheet_material = backsheet_material
        self.frame_material = frame_material

        logger.info(
            f"Initialized FireSafetyClassification for module {module_id}, "
            f"area={module_area_m2:.2f}m², roof_mounting={roof_mounting}"
        )

    def spread_of_flame_test(
        self,
        flame_exposure_time_min: float = 10.0,
        slope_deg: float = 7.0,
    ) -> SpreadOfFlameTestResult:
        """Perform spread of flame test per UL 790 / IEC 61730-2 Annex C.

        Tests the module's resistance to flame spread across its surface when
        exposed to a gas flame. This simulates external fire exposure conditions.

        Per UL 790:
        - Class A: No spread beyond 6 feet
        - Class B: No spread beyond 8 feet
        - Class C: No spread beyond 13 feet
        - Exposure time: 10 minutes for Class A, 4 min for B, 4 min for C

        Args:
            flame_exposure_time_min: Duration of flame exposure in minutes (default: 10).
            slope_deg: Roof slope in degrees (default: 7° ≈ 2:12 pitch).

        Returns:
            SpreadOfFlameTestResult containing flame spread distance and observations.
        """
        logger.info(
            f"Starting spread of flame test for module {self.module_id}: "
            f"{flame_exposure_time_min} min exposure"
        )

        # Simulate flame spread behavior based on materials
        # Flame spread depends on:
        # 1. Backsheet flammability
        # 2. Encapsulant type (EVA is more flammable than POE)
        # 3. Frame material (aluminum won't burn, but plastic can)
        # 4. Module sealing and edge protection

        # Material flammability factors
        backsheet_flammability = self._get_backsheet_flammability()
        frame_flammability = self._get_frame_flammability()

        # Base flame spread rate (cm/min)
        # Good materials: 1-5 cm/min
        # Moderate materials: 5-15 cm/min
        # Poor materials: >15 cm/min
        base_spread_rate_cm_min = (
            10.0 * backsheet_flammability * frame_flammability
        )

        # Exposure time factor (longer exposure = more spread)
        time_factor = np.sqrt(flame_exposure_time_min / 10.0)

        # Slope factor (steeper slope = faster spread)
        slope_factor = 1.0 + (slope_deg / 45.0)

        # Calculate total flame spread
        flame_spread_distance_cm = (
            base_spread_rate_cm_min * flame_exposure_time_min *
            time_factor * slope_factor
        )

        # Add random variation
        flame_spread_distance_cm *= np.random.normal(1.0, 0.15)
        flame_spread_distance_cm = max(0.0, flame_spread_distance_cm)

        # Determine if sustained flaming occurred
        # More likely with flammable materials and longer exposure
        sustained_flaming_probability = (
            0.1 * backsheet_flammability * (flame_exposure_time_min / 10.0)
        )
        sustained_flaming_observed = (
            np.random.random() < sustained_flaming_probability
        )

        # Check for roof deck penetration (critical failure)
        # Very unlikely with modern PV modules
        penetration_probability = 0.01 * backsheet_flammability
        roof_deck_penetration = np.random.random() < penetration_probability

        status = TestStatus.PASSED if not roof_deck_penetration else TestStatus.FAILED

        result = SpreadOfFlameTestResult(
            flame_spread_distance_cm=flame_spread_distance_cm,
            flame_exposure_time_min=flame_exposure_time_min,
            sustained_flaming_observed=sustained_flaming_observed,
            roof_deck_penetration=roof_deck_penetration,
            status=status,
        )

        logger.info(
            f"Spread of flame test complete: "
            f"Spread={flame_spread_distance_cm:.1f}cm, "
            f"Sustained flaming={'Yes' if sustained_flaming_observed else 'No'} - "
            f"{'PASS' if not roof_deck_penetration else 'FAIL'}"
        )

        return result

    def fire_penetration_test(
        self,
        test_duration_min: float = 90.0,
    ) -> FirePenetrationTestResult:
        """Perform fire penetration test per UL 790 / IEC 61730-2 Annex C.

        Tests the module's resistance to fire penetration through to the roof deck.
        This simulates a severe fire exposure from below the module.

        Per UL 790:
        - Class A: No burn-through for 90 minutes
        - Class B: No burn-through for 60 minutes
        - Class C: No burn-through for 20 minutes

        Args:
            test_duration_min: Test duration in minutes (default: 90 for Class A).

        Returns:
            FirePenetrationTestResult containing burn-through status.
        """
        logger.info(
            f"Starting fire penetration test for module {self.module_id}: "
            f"{test_duration_min} min duration"
        )

        # Simulate fire penetration resistance
        # Penetration time depends on:
        # 1. Backsheet thickness and material
        # 2. Encapsulant thickness
        # 3. Glass thickness (front glass is main barrier)
        # 4. Frame design and heat dissipation

        # Material resistance factors
        backsheet_resistance = 1.0 / self._get_backsheet_flammability()

        # Glass provides excellent fire barrier
        glass_resistance_factor = 10.0

        # Estimate time to burn-through (minutes)
        # Well-designed modules with glass: 60-120+ minutes
        # Modules with thin/flammable backsheet: 20-60 minutes
        time_to_burnthrough_min = (
            30.0 * backsheet_resistance * glass_resistance_factor *
            np.random.normal(1.0, 0.2)
        )

        # Determine if burn-through occurred during test
        burn_through_occurred = test_duration_min >= time_to_burnthrough_min

        # Check for roof deck damage (even without complete burn-through)
        # Damage can occur from heat transfer
        if burn_through_occurred:
            roof_deck_damage = True
        else:
            # Probability of damage increases with test duration
            damage_probability = (test_duration_min / time_to_burnthrough_min) * 0.1
            roof_deck_damage = np.random.random() < damage_probability

        status = TestStatus.PASSED if not burn_through_occurred else TestStatus.FAILED

        result = FirePenetrationTestResult(
            burn_through_occurred=burn_through_occurred,
            test_duration_min=test_duration_min,
            roof_deck_damage=roof_deck_damage,
            status=status,
        )

        logger.info(
            f"Fire penetration test complete: "
            f"{'Burn-through' if burn_through_occurred else 'No burn-through'} "
            f"at {test_duration_min} min - "
            f"{'PASS' if not burn_through_occurred else 'FAIL'}"
        )

        return result

    def fire_brand_test(
        self,
        brand_size_class: str = "A",
    ) -> FireBrandTestResult:
        """Perform flying brand (burning brand) test per UL 790 / IEC 61730-2.

        Tests the module's resistance to ignition from burning embers or brands
        that might land on the module surface during a wildfire or nearby fire.

        Per UL 790:
        - Class A: 12" x 12" burning brand
        - Class B: 6" x 6" burning brand
        - Class C: 1.5" diameter brand
        - Pass: No sustained flaming or ignition

        Args:
            brand_size_class: Brand size class ("A", "B", or "C").

        Returns:
            FireBrandTestResult containing ignition and burning status.
        """
        logger.info(
            f"Starting fire brand test for module {self.module_id}: "
            f"Brand class {brand_size_class}"
        )

        if brand_size_class not in ["A", "B", "C"]:
            raise ValueError(f"Invalid brand size class: {brand_size_class}")

        # Brand characteristics
        brand_sizes = {
            "A": (30.5, 30.5),  # 12" x 12" in cm
            "B": (15.2, 15.2),  # 6" x 6" in cm
            "C": (3.8, 3.8),    # 1.5" diameter in cm
        }

        brand_size_cm = brand_sizes[brand_size_class]

        # Simulate brand exposure
        # Ignition depends on:
        # 1. Surface material flammability
        # 2. Brand size (larger = more heat)
        # 3. Brand duration
        # 4. Surface temperature

        # Material ignitability
        backsheet_ignition_temp_c = self._get_backsheet_ignition_temp()

        # Brand heat transfer (larger brands deliver more heat)
        brand_area_cm2 = brand_size_cm[0] * brand_size_cm[1]
        heat_flux_factor = brand_area_cm2 / 100.0  # Normalized

        # Estimate surface temperature rise from brand
        surface_temp_rise_c = 100.0 * heat_flux_factor * np.random.normal(1.0, 0.2)

        # Ambient temperature
        ambient_temp_c = 25.0
        peak_surface_temp_c = ambient_temp_c + surface_temp_rise_c

        # Check for ignition
        ignition_occurred = peak_surface_temp_c >= backsheet_ignition_temp_c

        # If ignition occurs, check for sustained burning
        sustained_burning = False
        if ignition_occurred:
            # Sustained burning depends on material and oxygen availability
            burning_probability = 0.3 * self._get_backsheet_flammability()
            sustained_burning = np.random.random() < burning_probability

        status = TestStatus.PASSED if not ignition_occurred else TestStatus.FAILED

        result = FireBrandTestResult(
            ignition_occurred=ignition_occurred,
            brand_size_class=brand_size_class,
            sustained_burning=sustained_burning,
            status=status,
        )

        logger.info(
            f"Fire brand test complete: "
            f"{'Ignition' if ignition_occurred else 'No ignition'}, "
            f"{'Sustained burning' if sustained_burning else 'No sustained burning'} - "
            f"{'PASS' if not ignition_occurred else 'FAIL'}"
        )

        return result

    def classify_fire_rating(
        self,
        spread_of_flame_result: SpreadOfFlameTestResult,
        fire_penetration_result: FirePenetrationTestResult,
        fire_brand_result: FireBrandTestResult,
    ) -> FireClass:
        """Determine fire classification based on test results.

        Classifies the module's fire rating (Class A, B, C, or Not Rated)
        based on the results of spread of flame, fire penetration, and
        flying brand tests per UL 790 criteria.

        Classification criteria:
        - Class A: Highest fire resistance
          * Flame spread < 183cm (6 ft)
          * No burn-through for 90 min
          * Passes 12"x12" brand test
        - Class B: Medium fire resistance
          * Flame spread < 244cm (8 ft)
          * No burn-through for 60 min
          * Passes 6"x6" brand test
        - Class C: Basic fire resistance
          * Flame spread < 396cm (13 ft)
          * No burn-through for 20 min
          * Passes 1.5" brand test
        - Not Rated: Does not meet minimum requirements

        Args:
            spread_of_flame_result: Results from spread of flame test.
            fire_penetration_result: Results from fire penetration test.
            fire_brand_result: Results from flying brand test.

        Returns:
            FireClass classification (A, B, C, or Not Rated).
        """
        logger.info(f"Classifying fire rating for module {self.module_id}")

        # Convert flame spread to cm if needed
        flame_spread_cm = spread_of_flame_result.flame_spread_distance_cm

        # Check if basic requirements are met
        if spread_of_flame_result.roof_deck_penetration:
            logger.warning("Roof deck penetration - Not Rated")
            return FireClass.NOT_RATED

        if fire_penetration_result.burn_through_occurred:
            logger.warning("Burn-through occurred - Not Rated")
            return FireClass.NOT_RATED

        # Class A criteria
        class_a_criteria = {
            "flame_spread": flame_spread_cm < 183.0,  # 6 feet
            "penetration_time": not fire_penetration_result.burn_through_occurred,
            "brand_test": not fire_brand_result.ignition_occurred,
            "brand_size": fire_brand_result.brand_size_class == "A",
        }

        if all(class_a_criteria.values()):
            logger.info("Module classified as Fire Class A")
            return FireClass.CLASS_A

        # Class B criteria
        class_b_criteria = {
            "flame_spread": flame_spread_cm < 244.0,  # 8 feet
            "penetration_time": not fire_penetration_result.burn_through_occurred,
            "brand_test": not fire_brand_result.ignition_occurred,
            "brand_size": fire_brand_result.brand_size_class in ["A", "B"],
        }

        if all(class_b_criteria.values()):
            logger.info("Module classified as Fire Class B")
            return FireClass.CLASS_B

        # Class C criteria
        class_c_criteria = {
            "flame_spread": flame_spread_cm < 396.0,  # 13 feet
            "penetration_time": not fire_penetration_result.burn_through_occurred,
            "brand_test": not fire_brand_result.ignition_occurred,
        }

        if all(class_c_criteria.values()):
            logger.info("Module classified as Fire Class C")
            return FireClass.CLASS_C

        # Does not meet minimum requirements
        logger.warning("Module does not meet minimum fire rating - Not Rated")
        return FireClass.NOT_RATED

    def roof_mounting_fire_safety(
        self,
    ) -> Tuple[bool, str]:
        """Assess special fire safety considerations for roof-mounted modules.

        Evaluates additional fire safety requirements for building-integrated
        or roof-mounted PV systems, including:
        - Clearance requirements
        - Firefighter access pathways
        - Rapid shutdown capabilities
        - Arc fault protection

        Returns:
            Tuple of (compliant: bool, recommendations: str).
        """
        logger.info(
            f"Assessing roof mounting fire safety for module {self.module_id}"
        )

        recommendations = []
        compliant = True

        if not self.roof_mounting:
            return True, "Module not intended for roof mounting - no special requirements"

        # Check clearance requirements
        # Most fire codes require pathways for firefighter access
        recommendations.append(
            "Ensure 3-foot clearance pathways per fire code requirements"
        )
        recommendations.append(
            "Maintain 18-inch ridge clearance for ventilation"
        )

        # Rapid shutdown requirements (per NEC 690.12)
        recommendations.append(
            "Implement rapid shutdown per NEC 690.12 to reduce fire hazard"
        )

        # Arc fault protection
        recommendations.append(
            "Install arc-fault circuit interrupters (AFCI) per NEC 690.11"
        )

        # Material considerations
        if "plastic" in self.frame_material.lower():
            recommendations.append(
                "WARNING: Plastic frame materials may increase fire risk"
            )
            compliant = False

        # Fire class recommendations
        recommendations.append(
            "Recommend Fire Class A rating for roof-mounted applications"
        )

        recommendations_str = "\n".join(f"  - {rec}" for rec in recommendations)

        logger.info(
            f"Roof mounting assessment complete: {'COMPLIANT' if compliant else 'NON-COMPLIANT'}"
        )

        return compliant, recommendations_str

    def run_all_fire_tests(
        self,
        target_fire_class: FireClass = FireClass.CLASS_A,
    ) -> FireSafetyTestResult:
        """Execute complete fire safety test suite.

        Runs all fire safety tests (spread of flame, fire penetration, flying brand)
        and determines fire classification.

        Args:
            target_fire_class: Target fire class to test for (default: Class A).

        Returns:
            FireSafetyTestResult containing all test results and classification.
        """
        logger.info(
            f"Running complete fire safety test suite for module {self.module_id}, "
            f"target class {target_fire_class}"
        )

        # Determine test parameters based on target class
        if target_fire_class == FireClass.CLASS_A:
            flame_exposure_min = 10.0
            penetration_duration_min = 90.0
            brand_class = "A"
        elif target_fire_class == FireClass.CLASS_B:
            flame_exposure_min = 4.0
            penetration_duration_min = 60.0
            brand_class = "B"
        else:  # Class C or Not Rated
            flame_exposure_min = 4.0
            penetration_duration_min = 20.0
            brand_class = "C"

        # Run tests
        spread_of_flame = self.spread_of_flame_test(
            flame_exposure_time_min=flame_exposure_min
        )

        fire_penetration = self.fire_penetration_test(
            test_duration_min=penetration_duration_min
        )

        fire_brand = self.fire_brand_test(brand_size_class=brand_class)

        # Classify based on results
        fire_classification = self.classify_fire_rating(
            spread_of_flame,
            fire_penetration,
            fire_brand,
        )

        # Create result
        result = FireSafetyTestResult(
            spread_of_flame=spread_of_flame,
            fire_penetration=fire_penetration,
            fire_brand=fire_brand,
            fire_classification=fire_classification,
        )

        logger.info(
            f"Fire safety tests complete: Classification = {fire_classification}"
        )

        return result

    # Helper methods for material properties

    def _get_backsheet_flammability(self) -> float:
        """Get backsheet flammability factor (0.1 = low, 1.0 = high).

        Returns:
            Flammability factor based on backsheet material.
        """
        flammability_map = {
            "PET": 0.4,      # Polyethylene terephthalate - moderate
            "PVF": 0.2,      # Polyvinyl fluoride (Tedlar) - low
            "PA": 0.3,       # Polyamide - low-moderate
            "TPE": 0.5,      # Thermoplastic elastomer - moderate-high
            "PPE": 0.3,      # Polyphenylene ether - low-moderate
            "FEVE": 0.15,    # Fluoropolymer - very low
        }

        # Parse backsheet material string
        for material, flammability in flammability_map.items():
            if material in self.backsheet_material.upper():
                return flammability

        # Default for unknown materials
        return 0.5

    def _get_frame_flammability(self) -> float:
        """Get frame flammability factor (0.1 = low, 1.0 = high).

        Returns:
            Flammability factor based on frame material.
        """
        if "aluminum" in self.frame_material.lower():
            return 0.1  # Aluminum is non-combustible
        elif "steel" in self.frame_material.lower():
            return 0.1  # Steel is non-combustible
        elif "plastic" in self.frame_material.lower():
            return 0.8  # Plastic frames are flammable
        else:
            return 0.5  # Unknown material

    def _get_backsheet_ignition_temp(self) -> float:
        """Get backsheet ignition temperature in Celsius.

        Returns:
            Ignition temperature based on backsheet material.
        """
        ignition_temp_map = {
            "PET": 450.0,
            "PVF": 500.0,
            "PA": 480.0,
            "TPE": 400.0,
            "PPE": 460.0,
            "FEVE": 550.0,
        }

        # Parse backsheet material string
        for material, temp in ignition_temp_map.items():
            if material in self.backsheet_material.upper():
                return temp

        # Default ignition temperature
        return 450.0
