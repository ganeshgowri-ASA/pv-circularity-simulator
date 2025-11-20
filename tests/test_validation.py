"""Tests for PV system validation module."""

import pytest
from datetime import datetime

from src.b05_system_validation import (
    CodeComplianceChecker,
    DocumentationGenerator,
    EngineeringCalculationVerifier,
    PerformanceValidator,
    SystemValidator,
)
from src.models.validation_models import (
    ComplianceStatus,
    PerformanceMetrics,
    SystemConfiguration,
    SystemType,
)


@pytest.fixture
def sample_config() -> SystemConfiguration:
    """Create sample system configuration for testing."""
    return SystemConfiguration(
        system_type=SystemType.COMMERCIAL,
        system_name="Test PV System",
        location="Test Location",
        capacity_kw=100.0,
        module_count=250,
        inverter_count=2,
        string_count=20,
        modules_per_string=12,
        system_voltage_vdc=600.0,
        max_voltage_voc=800.0,
        operating_voltage_vmp=650.0,
        max_current_isc=10.0,
        operating_current_imp=9.5,
        ambient_temp_min=-10.0,
        ambient_temp_max=45.0,
        wind_speed_max=40.0,
        snow_load=50.0,
        jurisdiction="Test City",
        applicable_codes=["NEC 2020", "IEC 60364"],
    )


@pytest.fixture
def sample_performance_metrics() -> PerformanceMetrics:
    """Create sample performance metrics for testing."""
    return PerformanceMetrics(
        annual_energy_yield_kwh=150000.0,
        specific_yield_kwh_kwp=1500.0,
        performance_ratio=0.82,
        capacity_factor=0.20,
        loss_temperature=5.0,
        loss_soiling=2.0,
        loss_shading=1.0,
        loss_mismatch=2.0,
        loss_wiring=2.0,
        loss_inverter=3.0,
        loss_degradation=0.5,
        total_losses=15.5,
        is_energy_yield_realistic=True,
        is_pr_in_range=True,
        is_loss_budget_valid=True,
    )


class TestCodeComplianceChecker:
    """Tests for CodeComplianceChecker class."""

    def test_nec_690_compliance(self, sample_config: SystemConfiguration) -> None:
        """Test NEC 690 compliance checking."""
        checker = CodeComplianceChecker(sample_config)
        results = checker.nec_690_compliance()

        assert len(results) > 0
        assert all(hasattr(r, "code_name") for r in results)
        assert all(hasattr(r, "status") for r in results)

    def test_iec_60364_compliance(self, sample_config: SystemConfiguration) -> None:
        """Test IEC 60364 compliance checking."""
        checker = CodeComplianceChecker(sample_config)
        results = checker.iec_60364_compliance()

        assert len(results) > 0
        assert all("IEC" in r.code_name for r in results)

    def test_building_code_checks(self, sample_config: SystemConfiguration) -> None:
        """Test building code compliance checking."""
        checker = CodeComplianceChecker(sample_config)
        results = checker.building_code_checks()

        assert len(results) > 0

    def test_fire_safety_compliance(self, sample_config: SystemConfiguration) -> None:
        """Test fire safety compliance checking."""
        checker = CodeComplianceChecker(sample_config)
        results = checker.fire_safety_compliance()

        assert len(results) > 0

    def test_voltage_limit_check(self, sample_config: SystemConfiguration) -> None:
        """Test that voltage limits are properly checked."""
        checker = CodeComplianceChecker(sample_config)
        results = checker.nec_690_compliance()

        # Find voltage check result
        voltage_check = next(
            (r for r in results if "690.7" in r.section),
            None
        )

        assert voltage_check is not None
        assert voltage_check.checked_value is not None


class TestEngineeringCalculationVerifier:
    """Tests for EngineeringCalculationVerifier class."""

    def test_string_calculations(self, sample_config: SystemConfiguration) -> None:
        """Test string voltage and current calculations."""
        verifier = EngineeringCalculationVerifier(sample_config)
        results = verifier.verify_string_calculations()

        assert len(results) > 0
        assert any("voltage" in r.calculation_type for r in results)
        assert any("current" in r.calculation_type for r in results)

    def test_voltage_drop_calculation(self, sample_config: SystemConfiguration) -> None:
        """Test voltage drop calculation."""
        verifier = EngineeringCalculationVerifier(sample_config)
        result = verifier.check_voltage_drop(
            current=50.0,
            distance=30.0,
            wire_gauge="10AWG",
        )

        assert result.calculation_type == "voltage_drop"
        assert result.calculated_value >= 0
        assert result.unit == "%"

    def test_short_circuit_validation(self, sample_config: SystemConfiguration) -> None:
        """Test short circuit current validation."""
        verifier = EngineeringCalculationVerifier(sample_config)
        result = verifier.validate_short_circuit(
            parallel_strings=20,
            string_isc=10.0,
        )

        assert result.calculation_type == "short_circuit_current"
        assert result.calculated_value > 0

    def test_grounding_verification(self, sample_config: SystemConfiguration) -> None:
        """Test grounding conductor sizing."""
        verifier = EngineeringCalculationVerifier(sample_config)
        results = verifier.verify_grounding()

        assert len(results) > 0
        assert any("grounding" in r.calculation_type for r in results)

    def test_overcurrent_protection(self, sample_config: SystemConfiguration) -> None:
        """Test OCPD sizing."""
        verifier = EngineeringCalculationVerifier(sample_config)
        result = verifier.confirm_overcurrent_protection(
            continuous_current=50.0
        )

        assert result.calculation_type == "overcurrent_protection"
        assert result.calculated_value > 0


class TestPerformanceValidator:
    """Tests for PerformanceValidator class."""

    def test_energy_yield_sanity_check(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test energy yield sanity check."""
        validator = PerformanceValidator(sample_config, sample_performance_metrics)
        result = validator.energy_yield_sanity_check(150000.0)

        assert isinstance(result, bool)

    def test_pr_range_validation(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test performance ratio validation."""
        validator = PerformanceValidator(sample_config, sample_performance_metrics)
        result = validator.pr_range_validation(0.82)

        assert isinstance(result, bool)

    def test_loss_budget_verification(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test loss budget verification."""
        validator = PerformanceValidator(sample_config, sample_performance_metrics)
        result = validator.loss_budget_verification()

        assert isinstance(result, bool)

    def test_benchmark_comparison(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test benchmark comparison."""
        validator = PerformanceValidator(sample_config, sample_performance_metrics)
        comparison = validator.compare_to_benchmarks()

        assert "performance_ratio" in comparison
        assert "capacity_factor" in comparison
        assert "specific_yield" in comparison

    def test_unrealistic_results_detection(
        self,
        sample_config: SystemConfiguration,
    ) -> None:
        """Test detection of unrealistic results."""
        # Create unrealistic metrics
        unrealistic_metrics = PerformanceMetrics(
            annual_energy_yield_kwh=10000000.0,  # Unrealistically high
            specific_yield_kwh_kwp=5000.0,
            performance_ratio=1.5,  # > 100%
            capacity_factor=0.20,
            loss_temperature=-5.0,  # Negative loss
            loss_soiling=2.0,
            loss_shading=1.0,
            loss_mismatch=2.0,
            loss_wiring=2.0,
            loss_inverter=3.0,
            loss_degradation=0.5,
            total_losses=15.5,
            is_energy_yield_realistic=False,
            is_pr_in_range=False,
            is_loss_budget_valid=False,
        )

        validator = PerformanceValidator(sample_config, unrealistic_metrics)
        issues = validator.flag_unrealistic_results()

        assert len(issues) > 0


class TestSystemValidator:
    """Tests for SystemValidator class."""

    def test_complete_validation(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test complete system validation."""
        validator = SystemValidator(sample_config, sample_performance_metrics)
        report = validator.validate_complete_design()

        assert report is not None
        assert report.report_id is not None
        assert report.overall_status is not None
        assert len(report.electrical_validation) > 0
        assert len(report.structural_validation) > 0

    def test_electrical_code_checks(self, sample_config: SystemConfiguration) -> None:
        """Test electrical code compliance checks."""
        validator = SystemValidator(sample_config)
        results = validator.check_electrical_codes()

        assert len(results) > 0
        assert any("NEC" in r.check_name for r in results)
        assert any("IEC" in r.check_name for r in results)

    def test_structural_requirements(self, sample_config: SystemConfiguration) -> None:
        """Test structural requirements validation."""
        validator = SystemValidator(sample_config)
        results = validator.verify_structural_requirements()

        assert len(results) > 0

    def test_performance_validation(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test performance metrics validation."""
        validator = SystemValidator(sample_config, sample_performance_metrics)
        results = validator.validate_performance_metrics()

        assert len(results) > 0

    def test_validation_summary(
        self,
        sample_config: SystemConfiguration,
        sample_performance_metrics: PerformanceMetrics
    ) -> None:
        """Test validation summary generation."""
        validator = SystemValidator(sample_config, sample_performance_metrics)
        validator.validate_complete_design()
        summary = validator.get_validation_summary()

        assert "report_id" in summary
        assert "overall_status" in summary
        assert "total_issues" in summary


class TestDocumentationGenerator:
    """Tests for DocumentationGenerator class."""

    def test_engineering_package_generation(
        self,
        sample_config: SystemConfiguration,
        tmp_path: any
    ) -> None:
        """Test engineering package PDF generation."""
        doc_gen = DocumentationGenerator(
            sample_config,
            output_dir=str(tmp_path)
        )
        filepath = doc_gen.generate_engineering_package()

        assert filepath is not None
        assert filepath.endswith(".pdf")

    def test_cad_export(
        self,
        sample_config: SystemConfiguration,
        tmp_path: any
    ) -> None:
        """Test CAD drawing export."""
        doc_gen = DocumentationGenerator(
            sample_config,
            output_dir=str(tmp_path)
        )
        filepath = doc_gen.export_cad_drawing()

        assert filepath is not None
        assert filepath.endswith(".dxf")

    def test_calculations_spreadsheet(
        self,
        sample_config: SystemConfiguration,
        tmp_path: any
    ) -> None:
        """Test calculations spreadsheet generation."""
        doc_gen = DocumentationGenerator(
            sample_config,
            output_dir=str(tmp_path)
        )
        filepath = doc_gen.create_calculations_spreadsheet()

        assert filepath is not None
        assert filepath.endswith(".xlsx")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
