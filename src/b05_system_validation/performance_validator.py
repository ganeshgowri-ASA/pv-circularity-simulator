"""Performance validator for PV systems.

This module provides comprehensive validation of PV system performance metrics,
including energy yield sanity checks, performance ratio validation, loss budget
verification, and comparison to industry benchmarks.
"""

from typing import Dict, List, Optional, Tuple

from src.models.validation_models import (
    IssueItem,
    IssueSeverity,
    PerformanceMetrics,
    SystemConfiguration,
    SystemType,
)


class PerformanceValidator:
    """Comprehensive performance validator for PV systems.

    Validates system performance metrics against industry benchmarks,
    performs sanity checks on energy yield calculations, validates
    performance ratio and loss budgets, and flags unrealistic results.

    Attributes:
        config: System configuration
        performance_metrics: Performance metrics to validate
        validation_issues: List of validation issues found
    """

    # Industry benchmark ranges for Performance Ratio by system type
    PR_BENCHMARKS: Dict[SystemType, Tuple[float, float]] = {
        SystemType.RESIDENTIAL: (0.75, 0.85),
        SystemType.COMMERCIAL: (0.78, 0.88),
        SystemType.UTILITY_SCALE: (0.80, 0.90),
        SystemType.GROUND_MOUNT: (0.80, 0.90),
        SystemType.ROOFTOP: (0.75, 0.85),
        SystemType.CARPORT: (0.76, 0.86),
        SystemType.FLOATING: (0.82, 0.92),
        SystemType.BIFACIAL: (0.82, 0.92),
    }

    # Specific yield benchmarks by location (kWh/kWp/year)
    # These are rough estimates - actual values depend on location
    SPECIFIC_YIELD_RANGES: Dict[str, Tuple[float, float]] = {
        "high_irradiance": (1600, 2200),  # Desert, equatorial regions
        "medium_irradiance": (1200, 1600),  # Moderate climates
        "low_irradiance": (800, 1200),  # Northern latitudes, cloudy regions
    }

    # Maximum reasonable loss percentages
    MAX_LOSS_LIMITS: Dict[str, float] = {
        "temperature": 15.0,
        "soiling": 8.0,
        "shading": 10.0,
        "mismatch": 3.0,
        "wiring": 3.0,
        "inverter": 5.0,
        "degradation": 2.0,
    }

    # Capacity factor ranges by system type
    CAPACITY_FACTOR_RANGES: Dict[SystemType, Tuple[float, float]] = {
        SystemType.RESIDENTIAL: (0.14, 0.22),
        SystemType.COMMERCIAL: (0.15, 0.23),
        SystemType.UTILITY_SCALE: (0.18, 0.28),
        SystemType.GROUND_MOUNT: (0.18, 0.28),
        SystemType.ROOFTOP: (0.14, 0.22),
        SystemType.CARPORT: (0.15, 0.23),
        SystemType.FLOATING: (0.20, 0.30),
        SystemType.BIFACIAL: (0.20, 0.30),
    }

    def __init__(
        self,
        config: SystemConfiguration,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ) -> None:
        """Initialize performance validator.

        Args:
            config: System configuration
            performance_metrics: Performance metrics to validate (optional)
        """
        self.config = config
        self.performance_metrics = performance_metrics
        self.validation_issues: List[IssueItem] = []

    def energy_yield_sanity_check(
        self,
        annual_energy_kwh: float,
        irradiance_class: str = "medium_irradiance",
    ) -> bool:
        """Perform sanity check on annual energy yield.

        Validates that the predicted energy yield is within reasonable
        bounds based on system capacity and location irradiance.

        Args:
            annual_energy_kwh: Predicted annual energy yield (kWh)
            irradiance_class: Location irradiance class (high/medium/low)

        Returns:
            True if energy yield passes sanity check, False otherwise
        """
        # Calculate specific yield
        specific_yield = annual_energy_kwh / self.config.capacity_kw

        # Get expected range for irradiance class
        if irradiance_class not in self.SPECIFIC_YIELD_RANGES:
            irradiance_class = "medium_irradiance"

        min_yield, max_yield = self.SPECIFIC_YIELD_RANGES[irradiance_class]

        # Check if within reasonable range
        is_valid = min_yield <= specific_yield <= max_yield

        if not is_valid:
            severity = IssueSeverity.ERROR
            if specific_yield < min_yield:
                message = (
                    f"Energy yield ({specific_yield:.0f} kWh/kWp) is below "
                    f"expected minimum ({min_yield} kWh/kWp) for {irradiance_class}. "
                    f"Check system design and loss assumptions."
                )
            else:
                message = (
                    f"Energy yield ({specific_yield:.0f} kWh/kWp) exceeds "
                    f"expected maximum ({max_yield} kWh/kWp) for {irradiance_class}. "
                    f"Verify irradiance data and performance calculations."
                )

            self.validation_issues.append(
                IssueItem(
                    severity=severity,
                    category="performance",
                    code="ENERGY_YIELD_001",
                    message=message,
                    location="System Performance",
                    recommendation=(
                        "Review energy simulation inputs including irradiance data, "
                        "module specifications, and loss assumptions."
                    ),
                    reference=f"Expected range: {min_yield}-{max_yield} kWh/kWp/year",
                )
            )
        else:
            self.validation_issues.append(
                IssueItem(
                    severity=IssueSeverity.INFO,
                    category="performance",
                    code="ENERGY_YIELD_PASS",
                    message=(
                        f"Energy yield ({specific_yield:.0f} kWh/kWp) is within "
                        f"expected range for {irradiance_class}."
                    ),
                    location="System Performance",
                )
            )

        return is_valid

    def pr_range_validation(self, performance_ratio: Optional[float] = None) -> bool:
        """Validate that Performance Ratio is within expected range.

        Args:
            performance_ratio: Performance ratio to validate (0-1)
                Uses self.performance_metrics if not provided

        Returns:
            True if PR is within expected range, False otherwise
        """
        if performance_ratio is None:
            if self.performance_metrics is None:
                raise ValueError("No performance metrics provided")
            pr = self.performance_metrics.performance_ratio
        else:
            pr = performance_ratio

        # Get expected PR range for system type
        min_pr, max_pr = self.PR_BENCHMARKS.get(
            self.config.system_type,
            (0.75, 0.85)  # Default range
        )

        is_valid = min_pr <= pr <= max_pr

        if not is_valid:
            severity = IssueSeverity.WARNING if (min_pr - 0.05) <= pr <= (max_pr + 0.05) else IssueSeverity.ERROR

            if pr < min_pr:
                message = (
                    f"Performance Ratio ({pr:.2%}) is below expected minimum "
                    f"({min_pr:.2%}) for {self.config.system_type.value} systems. "
                    f"High losses or design issues may be present."
                )
                recommendation = (
                    "Review loss budget, check for excessive shading, soiling, "
                    "temperature losses, or equipment inefficiencies."
                )
            else:
                message = (
                    f"Performance Ratio ({pr:.2%}) exceeds expected maximum "
                    f"({max_pr:.2%}) for {self.config.system_type.value} systems. "
                    f"Verify calculation methodology."
                )
                recommendation = (
                    "Verify PR calculation includes all system losses. "
                    "Check irradiance data and reference conditions."
                )

            self.validation_issues.append(
                IssueItem(
                    severity=severity,
                    category="performance",
                    code="PR_RANGE_001",
                    message=message,
                    location="Performance Ratio",
                    recommendation=recommendation,
                    reference=f"Expected range: {min_pr:.2%}-{max_pr:.2%}",
                )
            )
        else:
            self.validation_issues.append(
                IssueItem(
                    severity=IssueSeverity.INFO,
                    category="performance",
                    code="PR_RANGE_PASS",
                    message=(
                        f"Performance Ratio ({pr:.2%}) is within expected range "
                        f"for {self.config.system_type.value} systems."
                    ),
                    location="Performance Ratio",
                )
            )

        return is_valid

    def loss_budget_verification(
        self,
        losses: Optional[Dict[str, float]] = None
    ) -> bool:
        """Verify that loss budget is reasonable and properly calculated.

        Args:
            losses: Dictionary of loss percentages by category
                Uses self.performance_metrics if not provided

        Returns:
            True if loss budget is valid, False otherwise
        """
        if losses is None:
            if self.performance_metrics is None:
                raise ValueError("No performance metrics provided")
            losses = {
                "temperature": self.performance_metrics.loss_temperature,
                "soiling": self.performance_metrics.loss_soiling,
                "shading": self.performance_metrics.loss_shading,
                "mismatch": self.performance_metrics.loss_mismatch,
                "wiring": self.performance_metrics.loss_wiring,
                "inverter": self.performance_metrics.loss_inverter,
                "degradation": self.performance_metrics.loss_degradation,
            }

        all_valid = True

        # Check individual loss categories against limits
        for loss_type, loss_value in losses.items():
            max_limit = self.MAX_LOSS_LIMITS.get(loss_type, 20.0)

            if loss_value > max_limit:
                all_valid = False
                self.validation_issues.append(
                    IssueItem(
                        severity=IssueSeverity.WARNING,
                        category="performance",
                        code=f"LOSS_{loss_type.upper()}",
                        message=(
                            f"{loss_type.capitalize()} loss ({loss_value:.1f}%) exceeds "
                            f"typical maximum ({max_limit:.1f}%). Verify assumptions."
                        ),
                        location=f"Loss Budget - {loss_type}",
                        recommendation=f"Review {loss_type} loss calculation and assumptions.",
                    )
                )

        # Check total losses
        total_losses = sum(losses.values())
        if total_losses > 40.0:
            all_valid = False
            self.validation_issues.append(
                IssueItem(
                    severity=IssueSeverity.ERROR,
                    category="performance",
                    code="LOSS_TOTAL_HIGH",
                    message=(
                        f"Total system losses ({total_losses:.1f}%) exceed 40%. "
                        f"This indicates potential design or calculation issues."
                    ),
                    location="Total Loss Budget",
                    recommendation=(
                        "Review all loss categories. Total losses > 40% typically "
                        "indicate errors or extremely suboptimal design."
                    ),
                )
            )
        elif total_losses < 10.0:
            all_valid = False
            self.validation_issues.append(
                IssueItem(
                    severity=IssueSeverity.WARNING,
                    category="performance",
                    code="LOSS_TOTAL_LOW",
                    message=(
                        f"Total system losses ({total_losses:.1f}%) are unusually low. "
                        f"Verify all loss categories are included."
                    ),
                    location="Total Loss Budget",
                    recommendation=(
                        "Ensure all loss categories are accounted for including "
                        "temperature, soiling, shading, mismatch, wiring, inverter, and degradation."
                    ),
                )
            )
        else:
            self.validation_issues.append(
                IssueItem(
                    severity=IssueSeverity.INFO,
                    category="performance",
                    code="LOSS_BUDGET_PASS",
                    message=f"Total system losses ({total_losses:.1f}%) are within reasonable range.",
                    location="Loss Budget",
                )
            )

        return all_valid

    def compare_to_benchmarks(
        self,
        metrics: Optional[PerformanceMetrics] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare system performance to industry benchmarks.

        Args:
            metrics: Performance metrics to compare (uses self.performance_metrics if not provided)

        Returns:
            Dictionary with benchmark comparisons
        """
        if metrics is None:
            if self.performance_metrics is None:
                raise ValueError("No performance metrics provided")
            metrics = self.performance_metrics

        comparisons: Dict[str, Dict[str, float]] = {}

        # Performance Ratio comparison
        pr_min, pr_max = self.PR_BENCHMARKS.get(self.config.system_type, (0.75, 0.85))
        pr_median = (pr_min + pr_max) / 2
        comparisons["performance_ratio"] = {
            "value": metrics.performance_ratio,
            "benchmark_min": pr_min,
            "benchmark_max": pr_max,
            "benchmark_median": pr_median,
            "difference_from_median": metrics.performance_ratio - pr_median,
            "percentile": self._calculate_percentile(
                metrics.performance_ratio, pr_min, pr_max
            ),
        }

        # Capacity Factor comparison
        cf_min, cf_max = self.CAPACITY_FACTOR_RANGES.get(
            self.config.system_type, (0.15, 0.25)
        )
        cf_median = (cf_min + cf_max) / 2
        comparisons["capacity_factor"] = {
            "value": metrics.capacity_factor,
            "benchmark_min": cf_min,
            "benchmark_max": cf_max,
            "benchmark_median": cf_median,
            "difference_from_median": metrics.capacity_factor - cf_median,
            "percentile": self._calculate_percentile(
                metrics.capacity_factor, cf_min, cf_max
            ),
        }

        # Specific Yield comparison (using medium irradiance as default)
        sy_min, sy_max = self.SPECIFIC_YIELD_RANGES["medium_irradiance"]
        sy_median = (sy_min + sy_max) / 2
        comparisons["specific_yield"] = {
            "value": metrics.specific_yield_kwh_kwp,
            "benchmark_min": sy_min,
            "benchmark_max": sy_max,
            "benchmark_median": sy_median,
            "difference_from_median": metrics.specific_yield_kwh_kwp - sy_median,
            "percentile": self._calculate_percentile(
                metrics.specific_yield_kwh_kwp, sy_min, sy_max
            ),
        }

        return comparisons

    def _calculate_percentile(self, value: float, min_val: float, max_val: float) -> float:
        """Calculate approximate percentile within range.

        Args:
            value: Value to assess
            min_val: Minimum of range
            max_val: Maximum of range

        Returns:
            Approximate percentile (0-100)
        """
        if value <= min_val:
            return 0.0
        elif value >= max_val:
            return 100.0
        else:
            return ((value - min_val) / (max_val - min_val)) * 100

    def flag_unrealistic_results(
        self,
        metrics: Optional[PerformanceMetrics] = None
    ) -> List[IssueItem]:
        """Flag unrealistic or suspicious performance results.

        Performs comprehensive checks to identify potentially erroneous
        or unrealistic performance predictions.

        Args:
            metrics: Performance metrics to check (uses self.performance_metrics if not provided)

        Returns:
            List of issues found with unrealistic results
        """
        if metrics is None:
            if self.performance_metrics is None:
                raise ValueError("No performance metrics provided")
            metrics = self.performance_metrics

        unrealistic_issues: List[IssueItem] = []

        # Check for unrealistically high energy yield
        max_theoretical_hours = 8760  # Hours per year
        theoretical_max_energy = self.config.capacity_kw * max_theoretical_hours
        if metrics.annual_energy_yield_kwh > theoretical_max_energy:
            unrealistic_issues.append(
                IssueItem(
                    severity=IssueSeverity.CRITICAL,
                    category="performance",
                    code="UNREALISTIC_ENERGY",
                    message=(
                        f"Annual energy ({metrics.annual_energy_yield_kwh:.0f} kWh) "
                        f"exceeds theoretical maximum ({theoretical_max_energy:.0f} kWh). "
                        f"This is physically impossible."
                    ),
                    location="Energy Yield",
                    recommendation="Check calculation methodology and input data.",
                )
            )

        # Check for Performance Ratio > 100%
        if metrics.performance_ratio > 1.0:
            unrealistic_issues.append(
                IssueItem(
                    severity=IssueSeverity.CRITICAL,
                    category="performance",
                    code="PR_EXCEEDS_100",
                    message=(
                        f"Performance Ratio ({metrics.performance_ratio:.2%}) exceeds 100%. "
                        f"This violates conservation of energy."
                    ),
                    location="Performance Ratio",
                    recommendation="Review PR calculation formula and reference conditions.",
                )
            )

        # Check for negative losses
        loss_fields = [
            ("temperature", metrics.loss_temperature),
            ("soiling", metrics.loss_soiling),
            ("shading", metrics.loss_shading),
            ("mismatch", metrics.loss_mismatch),
            ("wiring", metrics.loss_wiring),
            ("inverter", metrics.loss_inverter),
            ("degradation", metrics.loss_degradation),
        ]

        for loss_name, loss_value in loss_fields:
            if loss_value < 0:
                unrealistic_issues.append(
                    IssueItem(
                        severity=IssueSeverity.ERROR,
                        category="performance",
                        code=f"NEGATIVE_LOSS_{loss_name.upper()}",
                        message=f"{loss_name.capitalize()} loss cannot be negative: {loss_value}%",
                        location=f"Loss Budget - {loss_name}",
                        recommendation="Verify loss calculation. Losses must be >= 0.",
                    )
                )

        # Check capacity factor against physical limits
        # Maximum theoretical CF is ~35% for fixed-tilt systems
        if metrics.capacity_factor > 0.35:
            unrealistic_issues.append(
                IssueItem(
                    severity=IssueSeverity.WARNING,
                    category="performance",
                    code="CF_VERY_HIGH",
                    message=(
                        f"Capacity Factor ({metrics.capacity_factor:.2%}) is very high. "
                        f"Verify for tracking systems or exceptional solar resources."
                    ),
                    location="Capacity Factor",
                    recommendation=(
                        "Capacity factors > 35% are rare for fixed-tilt systems. "
                        "Verify system configuration and location."
                    ),
                )
            )

        return unrealistic_issues

    def validate_all(
        self,
        metrics: Optional[PerformanceMetrics] = None,
        irradiance_class: str = "medium_irradiance",
    ) -> Tuple[bool, List[IssueItem]]:
        """Perform all performance validation checks.

        Args:
            metrics: Performance metrics to validate
            irradiance_class: Location irradiance classification

        Returns:
            Tuple of (overall_valid, list of all issues)
        """
        if metrics:
            self.performance_metrics = metrics

        if self.performance_metrics is None:
            raise ValueError("No performance metrics provided")

        # Clear previous issues
        self.validation_issues = []

        # Run all validation checks
        energy_valid = self.energy_yield_sanity_check(
            self.performance_metrics.annual_energy_yield_kwh,
            irradiance_class
        )

        pr_valid = self.pr_range_validation()

        loss_valid = self.loss_budget_verification()

        # Check for unrealistic results
        unrealistic = self.flag_unrealistic_results()
        self.validation_issues.extend(unrealistic)

        # Overall validation status
        overall_valid = (
            energy_valid and
            pr_valid and
            loss_valid and
            len(unrealistic) == 0
        )

        return overall_valid, self.validation_issues

    def get_validation_summary(self) -> Dict[str, int]:
        """Get summary of validation results.

        Returns:
            Dictionary with issue counts by severity
        """
        return {
            "total_issues": len(self.validation_issues),
            "critical": sum(1 for i in self.validation_issues if i.severity == IssueSeverity.CRITICAL),
            "errors": sum(1 for i in self.validation_issues if i.severity == IssueSeverity.ERROR),
            "warnings": sum(1 for i in self.validation_issues if i.severity == IssueSeverity.WARNING),
            "info": sum(1 for i in self.validation_issues if i.severity == IssueSeverity.INFO),
        }
