"""
IEC 63202 CTM (Cell-to-Module) Power Loss Testing - BATCH4-B04-S03

This module provides functionality for IEC 63202 CTM power loss analysis including
optical, electrical, thermal, and mismatch loss characterization.
"""

import logging
from datetime import datetime
from typing import List, Tuple

import numpy as np

from src.iec_testing.models.test_models import (
    CTMLossBreakdown,
    IEC63202Result,
    IECStandard,
    IVCurveData,
    TestStatus,
)

logger = logging.getLogger(__name__)


class IEC63202Tester:
    """
    IEC 63202 CTM Power Loss Tester.

    Analyzes cell-to-module power loss according to IEC TS 63202:2020.
    """

    def __init__(self, test_lab: str = "Default Test Lab") -> None:
        """
        Initialize IEC 63202 tester.

        Args:
            test_lab: Name of the testing laboratory
        """
        self.test_lab = test_lab
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def measure_cell_performance(
        self, cell_id: str, cell_power: float = 6.8
    ) -> IVCurveData:
        """
        Measure individual solar cell performance at STC.

        Args:
            cell_id: Cell identifier
            cell_power: Nominal cell power in watts

        Returns:
            IVCurveData: Cell IV curve data
        """
        self.logger.info(f"Measuring cell performance for {cell_id}")

        # Generate realistic cell IV curve
        voltage = np.linspace(0, 0.68, 100).tolist()
        voc = 0.68
        isc = 12.5
        vmp = 0.57
        imp = 11.9
        pmax = vmp * imp

        current = [
            isc * (1 - (v / voc) ** 2.5) if v < voc else 0.0 for v in voltage
        ]

        return IVCurveData(
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
            efficiency=22.8,
        )

    def measure_module_performance(
        self, module_id: str, num_cells: int = 60
    ) -> IVCurveData:
        """
        Measure module performance at STC.

        Args:
            module_id: Module identifier
            num_cells: Number of cells in module

        Returns:
            IVCurveData: Module IV curve data
        """
        self.logger.info(f"Measuring module performance for {module_id}")

        # Generate realistic module IV curve
        voltage = np.linspace(0, 41, 100).tolist()
        voc = 41.0
        isc = 12.0  # Slightly less than cell due to optical losses
        vmp = 34.0
        imp = 11.5  # Slightly less due to mismatch
        pmax = vmp * imp

        current = [
            isc * (1 - (v / voc) ** 2.5) if v < voc else 0.0 for v in voltage
        ]

        return IVCurveData(
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

    def analyze_optical_losses(
        self, cell_isc: float, module_isc: float, num_cells: int
    ) -> float:
        """
        Analyze optical losses (reflection, absorption in encapsulant/glass).

        Args:
            cell_isc: Average cell short-circuit current
            module_isc: Module short-circuit current
            num_cells: Number of cells

        Returns:
            float: Optical loss percentage
        """
        self.logger.info("Analyzing optical losses")

        # Optical loss affects current
        expected_module_isc = cell_isc  # Ideally same as cell
        actual_loss = ((expected_module_isc - module_isc) / expected_module_isc) * 100

        return max(0, actual_loss)

    def analyze_electrical_losses(
        self, cell_voltage: float, module_voltage: float, num_cells: int
    ) -> float:
        """
        Analyze electrical losses (resistive losses in interconnects).

        Args:
            cell_voltage: Average cell voltage
            module_voltage: Module voltage
            num_cells: Number of cells

        Returns:
            float: Electrical loss percentage
        """
        self.logger.info("Analyzing electrical losses")

        # Electrical losses reduce voltage
        expected_module_voltage = cell_voltage * num_cells
        voltage_loss = ((expected_module_voltage - module_voltage) / expected_module_voltage) * 100

        return max(0, voltage_loss)

    def calculate_ctm_loss_breakdown(
        self,
        cell_iv_curves: List[IVCurveData],
        module_iv_curve: IVCurveData,
        module_area: float = 1.95,  # m²
    ) -> CTMLossBreakdown:
        """
        Calculate detailed CTM loss breakdown.

        Args:
            cell_iv_curves: List of individual cell IV curves
            module_iv_curve: Module IV curve
            module_area: Module area in m²

        Returns:
            CTMLossBreakdown: Detailed loss breakdown
        """
        self.logger.info("Calculating CTM loss breakdown")

        num_cells = len(cell_iv_curves)
        avg_cell_power = np.mean([curve.pmax for curve in cell_iv_curves])
        total_cell_power = avg_cell_power * num_cells
        module_power = module_iv_curve.pmax

        # Calculate individual loss components
        optical_loss = self.analyze_optical_losses(
            np.mean([c.isc for c in cell_iv_curves]),
            module_iv_curve.isc,
            num_cells,
        )

        electrical_loss = self.analyze_electrical_losses(
            np.mean([c.vmp for c in cell_iv_curves]),
            module_iv_curve.vmp,
            num_cells,
        )

        # Estimate other losses
        thermal_loss = 0.5  # Estimated thermal mismatch
        mismatch_loss = 1.2  # Cell-to-cell variation
        interconnection_loss = 1.5  # Interconnect resistance
        inactive_area_loss = 2.0  # Frame, gaps, etc.

        total_ctm_loss = (
            optical_loss
            + electrical_loss
            + thermal_loss
            + mismatch_loss
            + interconnection_loss
            + inactive_area_loss
        )

        ctm_ratio = module_power / total_cell_power

        return CTMLossBreakdown(
            optical_loss=optical_loss,
            electrical_loss=electrical_loss,
            thermal_loss=thermal_loss,
            mismatch_loss=mismatch_loss,
            interconnection_loss=interconnection_loss,
            inactive_area_loss=inactive_area_loss,
            total_ctm_loss=total_ctm_loss,
            ctm_ratio=ctm_ratio,
        )

    def run_full_ctm_analysis(
        self,
        module_id: str,
        module_type: str,
        manufacturer: str,
        test_campaign_id: str,
        num_cells: int = 60,
    ) -> IEC63202Result:
        """
        Run complete IEC 63202 CTM power loss analysis.

        Args:
            module_id: Module identifier
            module_type: Module type/model
            manufacturer: Module manufacturer
            test_campaign_id: Test campaign identifier
            num_cells: Number of cells in module

        Returns:
            IEC63202Result: Complete CTM analysis results
        """
        self.logger.info(f"Starting IEC 63202 CTM analysis for {module_id}")

        # Measure all cells
        cell_iv_curves = []
        for i in range(num_cells):
            cell_curve = self.measure_cell_performance(f"{module_id}_cell_{i}")
            cell_iv_curves.append(cell_curve)

        # Measure module
        module_iv_curve = self.measure_module_performance(module_id, num_cells)

        # Calculate losses
        ctm_loss_breakdown = self.calculate_ctm_loss_breakdown(
            cell_iv_curves, module_iv_curve
        )

        # Calculate performance metrics
        avg_cell_power = np.mean([curve.pmax for curve in cell_iv_curves])
        avg_cell_efficiency = np.mean([curve.efficiency for curve in cell_iv_curves])

        # Determine overall status
        # CTM ratio should typically be > 0.92 (< 8% loss)
        overall_status = (
            TestStatus.PASSED
            if ctm_loss_breakdown.ctm_ratio >= 0.92
            else TestStatus.CONDITIONAL_PASS
        )

        return IEC63202Result(
            test_campaign_id=test_campaign_id,
            module_type=module_type,
            manufacturer=manufacturer,
            test_lab=self.test_lab,
            test_date=datetime.now(),
            cell_power_avg=avg_cell_power,
            cell_efficiency_avg=avg_cell_efficiency,
            module_power=module_iv_curve.pmax,
            module_efficiency=module_iv_curve.efficiency,
            ctm_loss_breakdown=ctm_loss_breakdown,
            cell_iv_curves=cell_iv_curves,
            module_iv_curve=module_iv_curve,
            overall_status=overall_status,
            test_report_number=f"IEC63202-{test_campaign_id}",
        )
