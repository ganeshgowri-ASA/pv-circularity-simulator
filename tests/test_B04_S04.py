"""
Unit tests for BATCH4-B04-S04 IEC Testing Results & Reporting Dashboard.

This module contains comprehensive test stubs for all reporting and dashboard
functionality including IECTestResultsManager, TestReportGenerator,
ComplianceVisualization, and CertificationWorkflow.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pytest

from src.iec_testing.B04_S01_iec61215 import IEC61215Tester
from src.iec_testing.B04_S02_iec61730 import IEC61730Tester
from src.iec_testing.B04_S03_iec63202 import IEC63202Tester
from src.iec_testing.B04_S04_reporting_dashboard import (
    CertificationWorkflow,
    ComplianceVisualization,
    IECTestResultsManager,
    TestReportGenerator,
)
from src.iec_testing.models.test_models import (
    CertificationBodyType,
    ComplianceReport,
    IEC61215Result,
    IEC61730Result,
    IEC63202Result,
    IECStandard,
    TestStatus,
)


@pytest.fixture
def sample_iec61215_result() -> IEC61215Result:
    """
    Create sample IEC 61215 test result for testing.

    Returns:
        IEC61215Result: Sample test result
    """
    tester = IEC61215Tester()
    return tester.run_full_qualification(
        module_id="TEST_MODULE_001",
        module_type="PV-400W-PERC",
        manufacturer="Test Solar Inc.",
        test_campaign_id="TC-TEST-001",
    )


@pytest.fixture
def sample_iec61730_result() -> IEC61730Result:
    """
    Create sample IEC 61730 test result for testing.

    Returns:
        IEC61730Result: Sample test result
    """
    tester = IEC61730Tester()
    return tester.run_full_safety_qualification(
        module_id="TEST_MODULE_001",
        module_type="PV-400W-PERC",
        manufacturer="Test Solar Inc.",
        test_campaign_id="TC-TEST-001",
    )


@pytest.fixture
def sample_iec63202_result() -> IEC63202Result:
    """
    Create sample IEC 63202 test result for testing.

    Returns:
        IEC63202Result: Sample test result
    """
    tester = IEC63202Tester()
    return tester.run_full_ctm_analysis(
        module_id="TEST_MODULE_001",
        module_type="PV-400W-PERC",
        manufacturer="Test Solar Inc.",
        test_campaign_id="TC-TEST-001",
    )


@pytest.fixture
def test_results_manager(tmp_path: Path) -> IECTestResultsManager:
    """
    Create test results manager instance.

    Args:
        tmp_path: Temporary directory path

    Returns:
        IECTestResultsManager: Manager instance
    """
    return IECTestResultsManager(data_dir=tmp_path)


@pytest.fixture
def report_generator() -> TestReportGenerator:
    """
    Create report generator instance.

    Returns:
        TestReportGenerator: Report generator instance
    """
    return TestReportGenerator(company_name="Test Lab Inc.")


@pytest.fixture
def visualization() -> ComplianceVisualization:
    """
    Create visualization instance.

    Returns:
        ComplianceVisualization: Visualization instance
    """
    return ComplianceVisualization()


@pytest.fixture
def certification_workflow(tmp_path: Path) -> CertificationWorkflow:
    """
    Create certification workflow instance.

    Args:
        tmp_path: Temporary directory path

    Returns:
        CertificationWorkflow: Workflow instance
    """
    return CertificationWorkflow(data_dir=tmp_path)


class TestIECTestResultsManager:
    """Test suite for IECTestResultsManager class."""

    def test_load_test_results(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        sample_iec61730_result: IEC61730Result,
        sample_iec63202_result: IEC63202Result,
    ) -> None:
        """
        Test loading test results from all IEC standards.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            sample_iec61730_result: Sample IEC 61730 result
            sample_iec63202_result: Sample IEC 63202 result
        """
        counts = test_results_manager.load_test_results(
            result_61215=sample_iec61215_result,
            result_61730=sample_iec61730_result,
            result_63202=sample_iec63202_result,
        )

        assert counts["iec_61215"] == 1
        assert counts["iec_61730"] == 1
        assert counts["iec_63202"] == 1
        assert len(test_results_manager.iec_61215_results) == 1
        assert len(test_results_manager.iec_61730_results) == 1
        assert len(test_results_manager.iec_63202_results) == 1

    def test_aggregate_compliance_status(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        sample_iec61730_result: IEC61730Result,
        sample_iec63202_result: IEC63202Result,
    ) -> None:
        """
        Test aggregating compliance status across all standards.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            sample_iec61730_result: Sample IEC 61730 result
            sample_iec63202_result: Sample IEC 63202 result
        """
        test_results_manager.load_test_results(
            result_61215=sample_iec61215_result,
            result_61730=sample_iec61730_result,
            result_63202=sample_iec63202_result,
        )

        status = test_results_manager.aggregate_compliance_status()

        assert "total_tests" in status
        assert "passed_tests" in status
        assert "failed_tests" in status
        assert "compliance_rate" in status
        assert "overall_status" in status
        assert status["total_tests"] == 3
        assert 0 <= status["compliance_rate"] <= 100

    def test_compare_to_standards(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test comparing results to IEC standards.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        comparison = test_results_manager.compare_to_standards(sample_iec61215_result)

        assert "standard" in comparison
        assert "requirements_met" in comparison
        assert "requirements_failed" in comparison
        assert "margin_of_safety" in comparison
        assert comparison["standard"] == "IEC 61215:2021"

    def test_generate_compliance_matrix(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        sample_iec61730_result: IEC61730Result,
        sample_iec63202_result: IEC63202Result,
    ) -> None:
        """
        Test generating compliance matrix.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            sample_iec61730_result: Sample IEC 61730 result
            sample_iec63202_result: Sample IEC 63202 result
        """
        test_results_manager.load_test_results(
            result_61215=sample_iec61215_result,
            result_61730=sample_iec61730_result,
            result_63202=sample_iec63202_result,
        )

        matrix = test_results_manager.generate_compliance_matrix()

        assert len(matrix.iec_61215_tests) > 0
        assert len(matrix.iec_61730_tests) > 0
        assert len(matrix.iec_63202_tests) > 0
        assert matrix.total_tests > 0
        assert matrix.passed_tests >= 0
        assert matrix.failed_tests >= 0
        assert 0 <= matrix.compliance_rate <= 100

    def test_export_test_package_json(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        tmp_path: Path,
    ) -> None:
        """
        Test exporting test package in JSON format.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            tmp_path: Temporary directory path
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)

        exported = test_results_manager.export_test_package(tmp_path, format="json")

        assert "iec_61215" in exported
        assert exported["iec_61215"].exists()
        assert exported["iec_61215"].suffix == ".json"

    def test_export_test_package_excel(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        tmp_path: Path,
    ) -> None:
        """
        Test exporting test package in Excel format.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            tmp_path: Temporary directory path
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)

        exported = test_results_manager.export_test_package(tmp_path, format="excel")

        assert "excel" in exported
        assert exported["excel"].exists()
        assert exported["excel"].suffix == ".xlsx"

    def test_track_test_history(
        self,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test tracking test history for trend analysis.

        Args:
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)

        history = test_results_manager.track_test_history(
            module_type=sample_iec61215_result.module_type
        )

        assert history.module_type == sample_iec61215_result.module_type
        assert len(history.test_campaigns) > 0
        assert len(history.power_output_history) > 0
        assert history.mean_power >= 0
        assert history.std_power >= 0


class TestTestReportGenerator:
    """Test suite for TestReportGenerator class."""

    def test_generate_iec61215_report(
        self,
        report_generator: TestReportGenerator,
        sample_iec61215_result: IEC61215Result,
        tmp_path: Path,
    ) -> None:
        """
        Test generating IEC 61215 PDF report.

        Args:
            report_generator: Report generator fixture
            sample_iec61215_result: Sample IEC 61215 result
            tmp_path: Temporary directory path
        """
        output_path = tmp_path / "iec_61215_report.pdf"

        result_path = report_generator.generate_iec61215_report(
            sample_iec61215_result, output_path
        )

        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0

    def test_generate_iec61730_report(
        self,
        report_generator: TestReportGenerator,
        sample_iec61730_result: IEC61730Result,
        tmp_path: Path,
    ) -> None:
        """
        Test generating IEC 61730 PDF report.

        Args:
            report_generator: Report generator fixture
            sample_iec61730_result: Sample IEC 61730 result
            tmp_path: Temporary directory path
        """
        output_path = tmp_path / "iec_61730_report.pdf"

        result_path = report_generator.generate_iec61730_report(
            sample_iec61730_result, output_path
        )

        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0

    def test_generate_iec63202_report(
        self,
        report_generator: TestReportGenerator,
        sample_iec63202_result: IEC63202Result,
        tmp_path: Path,
    ) -> None:
        """
        Test generating IEC 63202 PDF report.

        Args:
            report_generator: Report generator fixture
            sample_iec63202_result: Sample IEC 63202 result
            tmp_path: Temporary directory path
        """
        output_path = tmp_path / "iec_63202_report.pdf"

        result_path = report_generator.generate_iec63202_report(
            sample_iec63202_result, output_path
        )

        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0

    def test_generate_combined_report(
        self,
        report_generator: TestReportGenerator,
        sample_iec61215_result: IEC61215Result,
        sample_iec61730_result: IEC61730Result,
        sample_iec63202_result: IEC63202Result,
        tmp_path: Path,
    ) -> None:
        """
        Test generating combined multi-standard report.

        Args:
            report_generator: Report generator fixture
            sample_iec61215_result: Sample IEC 61215 result
            sample_iec61730_result: Sample IEC 61730 result
            sample_iec63202_result: Sample IEC 63202 result
            tmp_path: Temporary directory path
        """
        output_path = tmp_path / "combined_report.pdf"

        result_path = report_generator.generate_combined_report(
            sample_iec61215_result,
            sample_iec61730_result,
            sample_iec63202_result,
            output_path,
        )

        assert result_path.exists()
        assert result_path.suffix == ".pdf"
        assert result_path.stat().st_size > 0


class TestComplianceVisualization:
    """Test suite for ComplianceVisualization class."""

    def test_pass_fail_summary(
        self,
        visualization: ComplianceVisualization,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test creating pass/fail summary chart.

        Args:
            visualization: Visualization fixture
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)
        matrix = test_results_manager.generate_compliance_matrix()

        fig = visualization.pass_fail_summary(matrix)

        assert fig is not None
        assert len(fig.data) > 0

    def test_degradation_timeline_chart(
        self,
        visualization: ComplianceVisualization,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test creating degradation timeline chart.

        Args:
            visualization: Visualization fixture
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)
        history = test_results_manager.track_test_history(
            sample_iec61215_result.module_type
        )

        fig = visualization.degradation_timeline_chart(history)

        assert fig is not None
        assert len(fig.data) > 0

    def test_iv_curve_comparison(
        self,
        visualization: ComplianceVisualization,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test creating IV curve comparison chart.

        Args:
            visualization: Visualization fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        fig = visualization.iv_curve_comparison(
            sample_iec61215_result.test_sequence.iv_curve_initial,
            sample_iec61215_result.test_sequence.iv_curve_final,
        )

        assert fig is not None
        assert len(fig.data) == 2

    def test_failure_mode_analysis(
        self,
        visualization: ComplianceVisualization,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
    ) -> None:
        """
        Test creating failure mode analysis chart.

        Args:
            visualization: Visualization fixture
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
        """
        test_results_manager.load_test_results(result_61215=sample_iec61215_result)
        matrix = test_results_manager.generate_compliance_matrix()

        fig = visualization.failure_mode_analysis(matrix)

        assert fig is not None

    def test_ctm_loss_waterfall(
        self,
        visualization: ComplianceVisualization,
        sample_iec63202_result: IEC63202Result,
    ) -> None:
        """
        Test creating CTM loss waterfall chart.

        Args:
            visualization: Visualization fixture
            sample_iec63202_result: Sample IEC 63202 result
        """
        fig = visualization.ctm_loss_waterfall(
            sample_iec63202_result.ctm_loss_breakdown
        )

        assert fig is not None
        assert len(fig.data) > 0


class TestCertificationWorkflow:
    """Test suite for CertificationWorkflow class."""

    def test_prepare_certification_package(
        self,
        certification_workflow: CertificationWorkflow,
        test_results_manager: IECTestResultsManager,
        sample_iec61215_result: IEC61215Result,
        sample_iec61730_result: IEC61730Result,
        sample_iec63202_result: IEC63202Result,
    ) -> None:
        """
        Test preparing certification package.

        Args:
            certification_workflow: Certification workflow fixture
            test_results_manager: Test results manager fixture
            sample_iec61215_result: Sample IEC 61215 result
            sample_iec61730_result: Sample IEC 61730 result
            sample_iec63202_result: Sample IEC 63202 result
        """
        test_results_manager.load_test_results(
            result_61215=sample_iec61215_result,
            result_61730=sample_iec61730_result,
            result_63202=sample_iec63202_result,
        )

        matrix = test_results_manager.generate_compliance_matrix()

        compliance_report = ComplianceReport(
            report_id="TEST-REPORT-001",
            module_type=sample_iec61215_result.module_type,
            manufacturer=sample_iec61215_result.manufacturer,
            iec_61215_result=sample_iec61215_result,
            iec_61730_result=sample_iec61730_result,
            iec_63202_result=sample_iec63202_result,
            compliance_matrix=matrix,
            overall_status=TestStatus.PASSED,
            certification_ready=True,
        )

        package = certification_workflow.prepare_certification_package(
            compliance_report=compliance_report,
            target_certifications=[CertificationBodyType.TUV_RHEINLAND, CertificationBodyType.UL],
            module_type=sample_iec61215_result.module_type,
            manufacturer=sample_iec61215_result.manufacturer,
        )

        assert package.package_id is not None
        assert package.module_type == sample_iec61215_result.module_type
        assert len(package.target_certifications) == 2

    def test_track_certification_status(
        self, certification_workflow: CertificationWorkflow
    ) -> None:
        """
        Test tracking certification status.

        Args:
            certification_workflow: Certification workflow fixture
        """
        statuses = certification_workflow.track_certification_status("TEST-PACKAGE-001")

        assert len(statuses) > 0
        assert all(hasattr(s, "certification_body") for s in statuses)
        assert all(hasattr(s, "status") for s in statuses)

    def test_manage_certification_costs(
        self, certification_workflow: CertificationWorkflow
    ) -> None:
        """
        Test managing certification costs.

        Args:
            certification_workflow: Certification workflow fixture
        """
        statuses = certification_workflow.track_certification_status("TEST-PACKAGE-001")
        costs = certification_workflow.manage_certification_costs(statuses)

        assert "total_cost" in costs
        assert costs["total_cost"] >= 0

    def test_international_certification_mapping(
        self, certification_workflow: CertificationWorkflow
    ) -> None:
        """
        Test mapping IEC standards to local certifications.

        Args:
            certification_workflow: Certification workflow fixture
        """
        mapping = certification_workflow.international_certification_mapping(
            IECStandard.IEC_61215
        )

        assert len(mapping) > 0
        assert "US" in mapping
        assert "Europe" in mapping
        assert "China" in mapping


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_test_and_report_workflow(
        self,
        test_results_manager: IECTestResultsManager,
        report_generator: TestReportGenerator,
        tmp_path: Path,
    ) -> None:
        """
        Test complete workflow from testing to reporting.

        Args:
            test_results_manager: Test results manager fixture
            report_generator: Report generator fixture
            tmp_path: Temporary directory path
        """
        # Run all tests
        tester_61215 = IEC61215Tester()
        result_61215 = tester_61215.run_full_qualification(
            module_id="INTEGRATION_TEST_001",
            module_type="PV-400W-PERC",
            manufacturer="Integration Test Inc.",
            test_campaign_id="TC-INTEGRATION-001",
        )

        tester_61730 = IEC61730Tester()
        result_61730 = tester_61730.run_full_safety_qualification(
            module_id="INTEGRATION_TEST_001",
            module_type="PV-400W-PERC",
            manufacturer="Integration Test Inc.",
            test_campaign_id="TC-INTEGRATION-001",
        )

        # Load results
        test_results_manager.load_test_results(
            result_61215=result_61215, result_61730=result_61730
        )

        # Generate compliance matrix
        matrix = test_results_manager.generate_compliance_matrix()
        assert matrix.total_tests > 0

        # Generate reports
        report_path_61215 = tmp_path / "integration_61215.pdf"
        report_generator.generate_iec61215_report(result_61215, report_path_61215)
        assert report_path_61215.exists()

        report_path_61730 = tmp_path / "integration_61730.pdf"
        report_generator.generate_iec61730_report(result_61730, report_path_61730)
        assert report_path_61730.exists()

        # Export data
        exported = test_results_manager.export_test_package(tmp_path, format="json")
        assert len(exported) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
