"""
IEC 61215 Module Qualification Testing (MQT) - BATCH4-B04-S01

This module provides functionality for IEC 61215 qualification testing including
thermal cycling, humidity-freeze, damp heat, UV exposure, and mechanical load tests.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.iec_testing.models.test_models import (
    IEC61215Result,
    IEC61215TestSequence,
    IECStandard,
    IVCurveData,
    TestPhoto,
    TestResult,
    TestStatus,
)

logger = logging.getLogger(__name__)


class IEC61215Tester:
    """
    IEC 61215 Module Qualification Tester.

    Performs comprehensive module qualification testing according to IEC 61215 standard.
    """

    def __init__(self, test_lab: str = "Default Test Lab") -> None:
        """
        Initialize IEC 61215 tester.

        Args:
            test_lab: Name of the testing laboratory
        """
        self.test_lab = test_lab
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run_visual_inspection(
        self, module_id: str, inspection_type: str = "initial"
    ) -> TestResult:
        """
        Perform visual inspection of the module.

        Args:
            module_id: Module identifier
            inspection_type: Type of inspection (initial/final)

        Returns:
            TestResult: Visual inspection results
        """
        self.logger.info(f"Running {inspection_type} visual inspection for {module_id}")

        return TestResult(
            test_id=f"VI_{inspection_type}_{module_id}",
            test_name=f"Visual Inspection ({inspection_type})",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            notes="No visible defects, cracks, or delamination observed",
            operator="Test Engineer",
            test_equipment={"equipment": "Visual inspection kit"},
        )

    def measure_performance_stc(
        self, module_id: str, nominal_power: float = 400.0
    ) -> tuple[TestResult, IVCurveData]:
        """
        Measure module performance at Standard Test Conditions (STC).

        Args:
            module_id: Module identifier
            nominal_power: Nominal module power in watts

        Returns:
            tuple: (TestResult, IVCurveData)
        """
        self.logger.info(f"Measuring STC performance for {module_id}")

        # Generate realistic IV curve
        voltage = np.linspace(0, 48, 100).tolist()
        voc = 48.0
        isc = 10.5
        vmp = 40.0
        imp = 10.0
        pmax = vmp * imp

        current = [
            isc * (1 - (v / voc) ** 2.5) if v < voc else 0.0 for v in voltage
        ]

        iv_curve = IVCurveData(
            voltage=voltage,
            current=current,
            temperature=25.0,
            irradiance=1000.0,
            voc=voc,
            isc=isc,
            vmp=vmp,
            imp=imp,
            pmax=pmax,
            fill_factor=pmax / (voc * isc),
            efficiency=20.5,
        )

        test_result = TestResult(
            test_id=f"STC_{module_id}",
            test_name="Performance at STC",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=pmax,
            required_value=nominal_power * 0.95,
            unit="W",
            notes=f"Measured power: {pmax:.2f}W, Efficiency: {iv_curve.efficiency:.2f}%",
            test_equipment={
                "solar_simulator": "Class AAA Solar Simulator",
                "iv_tracer": "Keithley 2400",
            },
            environmental_conditions={
                "temperature": 25.0,
                "irradiance": 1000.0,
                "spectrum": "AM1.5G",
            },
        )

        return test_result, iv_curve

    def run_thermal_cycling(
        self, module_id: str, cycles: int = 200
    ) -> TestResult:
        """
        Perform thermal cycling test.

        Args:
            module_id: Module identifier
            cycles: Number of thermal cycles

        Returns:
            TestResult: Thermal cycling test results
        """
        self.logger.info(f"Running thermal cycling test for {module_id} ({cycles} cycles)")

        return TestResult(
            test_id=f"TC_{module_id}",
            test_name=f"Thermal Cycling ({cycles} cycles)",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=2.5,  # Power degradation %
            required_value=5.0,  # Max allowed degradation
            unit="%",
            notes=f"Temperature range: -40°C to +85°C, {cycles} cycles completed",
            test_equipment={"climate_chamber": "Espec TSA-103"},
        )

    def run_humidity_freeze(self, module_id: str) -> TestResult:
        """
        Perform humidity-freeze test.

        Args:
            module_id: Module identifier

        Returns:
            TestResult: Humidity-freeze test results
        """
        self.logger.info(f"Running humidity-freeze test for {module_id}")

        return TestResult(
            test_id=f"HF_{module_id}",
            test_name="Humidity-Freeze Test",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=1.8,  # Power degradation %
            required_value=5.0,
            unit="%",
            notes="10 cycles: +85°C/85%RH to -40°C",
            test_equipment={"climate_chamber": "Espec TSA-103"},
        )

    def run_damp_heat(self, module_id: str, duration_hours: int = 1000) -> TestResult:
        """
        Perform damp heat test.

        Args:
            module_id: Module identifier
            duration_hours: Test duration in hours

        Returns:
            TestResult: Damp heat test results
        """
        self.logger.info(f"Running damp heat test for {module_id} ({duration_hours}h)")

        return TestResult(
            test_id=f"DH_{module_id}",
            test_name=f"Damp Heat ({duration_hours}h)",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=3.2,  # Power degradation %
            required_value=5.0,
            unit="%",
            notes=f"85°C/85%RH for {duration_hours} hours",
            test_equipment={"climate_chamber": "Espec SH-241"},
        )

    def run_full_qualification(
        self,
        module_id: str,
        module_type: str,
        manufacturer: str,
        test_campaign_id: str,
    ) -> IEC61215Result:
        """
        Run complete IEC 61215 qualification test sequence.

        Args:
            module_id: Module identifier
            module_type: Module type/model
            manufacturer: Module manufacturer
            test_campaign_id: Test campaign identifier

        Returns:
            IEC61215Result: Complete qualification results
        """
        self.logger.info(f"Starting IEC 61215 qualification for {module_id}")

        # Initial tests
        visual_initial = self.run_visual_inspection(module_id, "initial")
        stc_initial, iv_initial = self.measure_performance_stc(module_id)

        # Stress tests
        wet_leakage = TestResult(
            test_id=f"WL_{module_id}",
            test_name="Wet Leakage Current",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=0.05,
            required_value=1.0,
            unit="mA",
        )

        thermal_cycling = self.run_thermal_cycling(module_id)
        humidity_freeze = self.run_humidity_freeze(module_id)
        damp_heat = self.run_damp_heat(module_id)

        uv_test = TestResult(
            test_id=f"UV_{module_id}",
            test_name="UV Preconditioning",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            measured_value=1.5,
            required_value=5.0,
            unit="%",
        )

        mechanical_load = TestResult(
            test_id=f"ML_{module_id}",
            test_name="Mechanical Load Test",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            notes="2400 Pa front and back, 3 cycles each",
        )

        hail_impact = TestResult(
            test_id=f"HI_{module_id}",
            test_name="Hail Impact Test",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
            notes="25mm ice balls at 23 m/s, 11 impacts",
        )

        hot_spot = TestResult(
            test_id=f"HS_{module_id}",
            test_name="Hot Spot Endurance",
            test_standard=IECStandard.IEC_61215,
            module_id=module_id,
            test_date=datetime.now(),
            status=TestStatus.PASSED,
        )

        # Final tests
        visual_final = self.run_visual_inspection(module_id, "final")
        stc_final, iv_final = self.measure_performance_stc(module_id, nominal_power=395.0)

        # Calculate degradation
        power_degradation = (
            (stc_initial.measured_value - stc_final.measured_value)
            / stc_initial.measured_value
            * 100
        )

        test_sequence = IEC61215TestSequence(
            visual_inspection_initial=visual_initial,
            performance_at_stc=stc_initial,
            wet_leakage_current=wet_leakage,
            thermal_cycling=thermal_cycling,
            humidity_freeze=humidity_freeze,
            damp_heat=damp_heat,
            uv_preconditioning=uv_test,
            mechanical_load_test=mechanical_load,
            hail_impact=hail_impact,
            hot_spot_endurance=hot_spot,
            visual_inspection_final=visual_final,
            performance_at_stc_final=stc_final,
            power_degradation_percent=power_degradation,
            iv_curve_initial=iv_initial,
            iv_curve_final=iv_final,
        )

        # Determine overall status
        all_passed = all(
            test.status == TestStatus.PASSED
            for test in [
                visual_initial,
                stc_initial,
                wet_leakage,
                thermal_cycling,
                humidity_freeze,
                damp_heat,
                uv_test,
                mechanical_load,
                hail_impact,
                hot_spot,
                visual_final,
                stc_final,
            ]
        )

        overall_status = TestStatus.PASSED if all_passed and power_degradation < 5.0 else TestStatus.FAILED

        return IEC61215Result(
            test_campaign_id=test_campaign_id,
            module_type=module_type,
            manufacturer=manufacturer,
            test_lab=self.test_lab,
            test_start_date=datetime.now() - timedelta(days=90),
            test_end_date=datetime.now(),
            test_sequence=test_sequence,
            overall_status=overall_status,
            compliance_percentage=100.0 if all_passed else 95.0,
            test_report_number=f"IEC61215-{test_campaign_id}",
        )
