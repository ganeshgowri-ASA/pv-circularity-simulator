"""
Application Suite Module (B13-B15)
===================================
Integrates:
- B13: Financial Analysis & Bankability Assessment
- B14: Core Infrastructure & Data Management
- B15: Main Application Integration Layer

This module provides financial modeling, data infrastructure,
and the main application integration framework.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import json


# ============================================================================
# B13: FINANCIAL ANALYSIS & BANKABILITY ASSESSMENT
# ============================================================================

class FinancingType(str, Enum):
    """Project financing types."""
    SELF_FINANCED = "Self-Financed"
    BANK_LOAN = "Bank Loan"
    PROJECT_FINANCE = "Project Finance"
    PPA = "Power Purchase Agreement"
    LEASE = "Lease"
    HYBRID = "Hybrid"


class IncentiveType(str, Enum):
    """Government incentive types."""
    ITC = "Investment Tax Credit"
    PTC = "Production Tax Credit"
    MACRS = "MACRS Depreciation"
    GRANT = "Grant"
    FEED_IN_TARIFF = "Feed-in Tariff"
    NET_METERING = "Net Metering"
    REC = "Renewable Energy Certificate"


class FinancialParameters(BaseModel):
    """Financial modeling parameters."""

    project_name: str = Field(..., description="Project name")
    total_capex_usd: float = Field(..., ge=0, description="Total capital expenditure ($)")
    annual_opex_usd: float = Field(..., ge=0, description="Annual operating expenditure ($)")
    electricity_price_kwh: float = Field(..., ge=0, description="Electricity price ($/kWh)")
    annual_energy_kwh: float = Field(..., ge=0, description="Annual energy production (kWh)")
    project_lifetime_years: int = Field(default=25, ge=1, le=50, description="Project lifetime (years)")
    discount_rate: float = Field(default=8.0, ge=0, le=30, description="Discount rate (%)")
    inflation_rate: float = Field(default=2.0, ge=0, le=10, description="Inflation rate (%)")
    degradation_rate: float = Field(default=0.5, ge=0, le=2, description="Annual degradation (%/year)")
    debt_equity_ratio: float = Field(default=70.0, ge=0, le=100, description="Debt financing (%)")
    interest_rate: float = Field(default=5.0, ge=0, le=20, description="Loan interest rate (%)")
    loan_term_years: int = Field(default=15, ge=1, le=30, description="Loan term (years)")
    tax_rate: float = Field(default=21.0, ge=0, le=50, description="Corporate tax rate (%)")

    class Config:
        use_enum_values = True


class FinancialResults(BaseModel):
    """Financial analysis results."""

    lcoe_usd_kwh: float = Field(..., description="Levelized Cost of Energy ($/kWh)")
    npv_usd: float = Field(..., description="Net Present Value ($)")
    irr_percentage: float = Field(..., description="Internal Rate of Return (%)")
    payback_period_years: float = Field(..., description="Simple payback period (years)")
    discounted_payback_years: float = Field(..., description="Discounted payback period (years)")
    equity_irr: float = Field(..., description="Equity IRR (%)")
    debt_service_coverage_ratio: float = Field(..., description="Average DSCR")
    annual_cash_flows: List[float] = Field(default_factory=list, description="Annual cash flows ($)")
    cumulative_cash_flows: List[float] = Field(default_factory=list, description="Cumulative cash flows ($)")
    bankability_score: float = Field(..., ge=0, le=100, description="Bankability score (0-100)")
    financial_viability: str = Field(..., description="Financial viability assessment")


class FinancialAnalyzer:
    """
    Financial Analysis & Bankability Assessment Engine.
    Comprehensive financial modeling for PV projects.
    """

    def __init__(self):
        """Initialize financial analyzer."""
        self.analyses: List[FinancialResults] = []

    def analyze_project(
        self,
        params: FinancialParameters,
        incentives: Optional[Dict[IncentiveType, float]] = None
    ) -> FinancialResults:
        """
        Perform comprehensive financial analysis.

        Args:
            params: Financial parameters
            incentives: Tax incentives and benefits

        Returns:
            Complete financial analysis results
        """
        # Calculate annual revenues and costs
        annual_revenues, annual_costs = self._calculate_annual_cashflows(params, incentives)

        # Calculate NPV and IRR
        npv = self._calculate_npv(annual_revenues, annual_costs, params.discount_rate)
        irr = self._calculate_irr(annual_revenues, annual_costs)

        # Calculate LCOE
        lcoe = self._calculate_lcoe(params, annual_costs)

        # Calculate payback periods
        simple_payback = self._calculate_simple_payback(params, annual_revenues, annual_costs)
        discounted_payback = self._calculate_discounted_payback(
            params, annual_revenues, annual_costs
        )

        # Calculate debt metrics
        debt_amount = params.total_capex_usd * (params.debt_equity_ratio / 100)
        dscr = self._calculate_dscr(
            annual_revenues, annual_costs, debt_amount, params.interest_rate, params.loan_term_years
        )

        # Calculate equity IRR
        equity_irr = self._calculate_equity_irr(
            params, annual_revenues, annual_costs, debt_amount
        )

        # Calculate bankability score
        bankability = self._calculate_bankability_score(npv, irr, dscr, simple_payback)

        # Determine viability
        viability = self._assess_viability(bankability, irr, npv)

        # Calculate cash flow profiles
        annual_cf = [r - c for r, c in zip(annual_revenues, annual_costs)]
        cumulative_cf = np.cumsum(annual_cf).tolist()

        results = FinancialResults(
            lcoe_usd_kwh=lcoe,
            npv_usd=npv,
            irr_percentage=irr,
            payback_period_years=simple_payback,
            discounted_payback_years=discounted_payback,
            equity_irr=equity_irr,
            debt_service_coverage_ratio=dscr,
            annual_cash_flows=annual_cf,
            cumulative_cash_flows=cumulative_cf,
            bankability_score=bankability,
            financial_viability=viability
        )

        self.analyses.append(results)
        return results

    def _calculate_annual_cashflows(
        self,
        params: FinancialParameters,
        incentives: Optional[Dict[IncentiveType, float]]
    ) -> Tuple[List[float], List[float]]:
        """Calculate annual revenues and costs."""
        revenues = []
        costs = []

        for year in range(params.project_lifetime_years):
            # Revenue (with degradation and inflation)
            degradation_factor = (1 - params.degradation_rate / 100) ** year
            energy_production = params.annual_energy_kwh * degradation_factor
            electricity_price = params.electricity_price_kwh * (1 + params.inflation_rate / 100) ** year
            annual_revenue = energy_production * electricity_price

            # Add incentives
            if incentives:
                if IncentiveType.PTC in incentives and year < 10:  # PTC typically 10 years
                    annual_revenue += energy_production * incentives[IncentiveType.PTC]
                if IncentiveType.FEED_IN_TARIFF in incentives:
                    annual_revenue += energy_production * incentives[IncentiveType.FEED_IN_TARIFF]

            revenues.append(annual_revenue)

            # Costs
            annual_opex = params.annual_opex_usd * (1 + params.inflation_rate / 100) ** year
            costs.append(annual_opex)

        # Apply ITC if present
        if incentives and IncentiveType.ITC in incentives:
            itc_benefit = params.total_capex_usd * incentives[IncentiveType.ITC]
            revenues[0] += itc_benefit

        return revenues, costs

    def _calculate_lcoe(self, params: FinancialParameters, annual_costs: List[float]) -> float:
        """Calculate Levelized Cost of Energy."""
        # Present value of costs
        total_costs = params.total_capex_usd
        for year, cost in enumerate(annual_costs, start=1):
            discount_factor = 1 / ((1 + params.discount_rate / 100) ** year)
            total_costs += cost * discount_factor

        # Present value of energy
        total_energy = 0
        for year in range(params.project_lifetime_years):
            degradation_factor = (1 - params.degradation_rate / 100) ** year
            energy = params.annual_energy_kwh * degradation_factor
            discount_factor = 1 / ((1 + params.discount_rate / 100) ** (year + 1))
            total_energy += energy * discount_factor

        return total_costs / total_energy if total_energy > 0 else 0

    def _calculate_npv(
        self,
        revenues: List[float],
        costs: List[float],
        discount_rate: float
    ) -> float:
        """Calculate Net Present Value."""
        npv = -sum(costs[:1]) if costs else 0  # Initial capex (negative)

        for year, (rev, cost) in enumerate(zip(revenues, costs), start=1):
            net_cf = rev - cost
            discount_factor = 1 / ((1 + discount_rate / 100) ** year)
            npv += net_cf * discount_factor

        return npv

    def _calculate_irr(self, revenues: List[float], costs: List[float]) -> float:
        """Calculate Internal Rate of Return."""
        cash_flows = [-costs[0]] if costs else [0]  # Initial investment
        cash_flows.extend([r - c for r, c in zip(revenues, costs)])

        # Newton-Raphson method for IRR
        irr = 0.1  # Initial guess
        for _ in range(100):  # Max iterations
            npv = sum(cf / ((1 + irr) ** i) for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / ((1 + irr) ** (i + 1)) for i, cf in enumerate(cash_flows))

            if abs(npv) < 1e-6:
                break

            if npv_derivative != 0:
                irr = irr - npv / npv_derivative

        return irr * 100  # Convert to percentage

    def _calculate_simple_payback(
        self,
        params: FinancialParameters,
        revenues: List[float],
        costs: List[float]
    ) -> float:
        """Calculate simple payback period."""
        cumulative = -params.total_capex_usd

        for year, (rev, cost) in enumerate(zip(revenues, costs), start=1):
            cumulative += (rev - cost)
            if cumulative >= 0:
                return year

        return params.project_lifetime_years  # Never paid back

    def _calculate_discounted_payback(
        self,
        params: FinancialParameters,
        revenues: List[float],
        costs: List[float]
    ) -> float:
        """Calculate discounted payback period."""
        cumulative = -params.total_capex_usd

        for year, (rev, cost) in enumerate(zip(revenues, costs), start=1):
            net_cf = rev - cost
            discount_factor = 1 / ((1 + params.discount_rate / 100) ** year)
            cumulative += net_cf * discount_factor

            if cumulative >= 0:
                return year

        return params.project_lifetime_years

    def _calculate_dscr(
        self,
        revenues: List[float],
        costs: List[float],
        debt_amount: float,
        interest_rate: float,
        loan_term: int
    ) -> float:
        """Calculate Debt Service Coverage Ratio."""
        if debt_amount == 0:
            return float('inf')

        # Calculate annual debt service (loan payment)
        r = interest_rate / 100
        annual_debt_service = debt_amount * (r * (1 + r) ** loan_term) / ((1 + r) ** loan_term - 1)

        # Calculate average DSCR
        dscrs = []
        for year in range(min(loan_term, len(revenues))):
            net_operating_income = revenues[year] - costs[year]
            dscr = net_operating_income / annual_debt_service if annual_debt_service > 0 else 0
            dscrs.append(dscr)

        return np.mean(dscrs) if dscrs else 0

    def _calculate_equity_irr(
        self,
        params: FinancialParameters,
        revenues: List[float],
        costs: List[float],
        debt_amount: float
    ) -> float:
        """Calculate Equity IRR."""
        equity_investment = params.total_capex_usd - debt_amount

        # Equity cash flows
        equity_cf = [-equity_investment]

        for year in range(params.project_lifetime_years):
            net_cf = revenues[year] - costs[year]
            # Subtract debt service
            if year < params.loan_term_years:
                r = params.interest_rate / 100
                debt_service = debt_amount * (r * (1 + r) ** params.loan_term_years) / ((1 + r) ** params.loan_term_years - 1)
                net_cf -= debt_service

            equity_cf.append(net_cf)

        # Calculate IRR on equity cash flows
        return self._calculate_irr_from_cashflows(equity_cf)

    def _calculate_irr_from_cashflows(self, cash_flows: List[float]) -> float:
        """Calculate IRR from cash flow list."""
        irr = 0.1
        for _ in range(100):
            npv = sum(cf / ((1 + irr) ** i) for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / ((1 + irr) ** (i + 1)) for i, cf in enumerate(cash_flows))

            if abs(npv) < 1e-6:
                break

            if npv_derivative != 0:
                irr = irr - npv / npv_derivative

        return irr * 100

    def _calculate_bankability_score(
        self,
        npv: float,
        irr: float,
        dscr: float,
        payback: float
    ) -> float:
        """Calculate bankability score (0-100)."""
        score = 0

        # NPV score (0-25 points)
        if npv > 0:
            score += min(25, npv / 100000 * 5)

        # IRR score (0-30 points)
        if irr > 8:
            score += min(30, (irr - 8) * 3)

        # DSCR score (0-25 points)
        if dscr > 1.0:
            score += min(25, (dscr - 1.0) * 25)

        # Payback score (0-20 points)
        if payback < 10:
            score += 20 - (payback * 2)

        return min(100, score)

    def _assess_viability(self, bankability: float, irr: float, npv: float) -> str:
        """Assess financial viability."""
        if bankability >= 80 and irr >= 12 and npv > 0:
            return "Highly Bankable - Excellent Investment"
        elif bankability >= 60 and irr >= 8 and npv > 0:
            return "Bankable - Good Investment"
        elif bankability >= 40 and irr >= 5:
            return "Marginally Viable - Further Analysis Needed"
        else:
            return "Not Viable - Investment Not Recommended"


# ============================================================================
# B14: CORE INFRASTRUCTURE & DATA MANAGEMENT
# ============================================================================

class DataSourceType(str, Enum):
    """Data source types."""
    SCADA = "SCADA"
    WEATHER_API = "Weather API"
    DATABASE = "Database"
    FILE_SYSTEM = "File System"
    EXTERNAL_API = "External API"
    MANUAL_INPUT = "Manual Input"


class DataQualityMetrics(BaseModel):
    """Data quality assessment metrics."""

    completeness: float = Field(..., ge=0, le=100, description="Data completeness (%)")
    accuracy: float = Field(..., ge=0, le=100, description="Data accuracy (%)")
    timeliness: float = Field(..., ge=0, le=100, description="Data timeliness (%)")
    consistency: float = Field(..., ge=0, le=100, description="Data consistency (%)")
    overall_quality: float = Field(..., ge=0, le=100, description="Overall quality score (%)")


class DataManager:
    """
    Core Infrastructure & Data Management System.
    Handles data collection, validation, storage, and retrieval.
    """

    def __init__(self):
        """Initialize data manager."""
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def ingest_data(
        self,
        data: pd.DataFrame,
        source_type: DataSourceType,
        data_id: str
    ) -> bool:
        """
        Ingest data from various sources.

        Args:
            data: Data to ingest
            source_type: Source type
            data_id: Data identifier

        Returns:
            Success status
        """
        # Validate data
        if data.empty:
            return False

        # Store data
        self.data_cache[data_id] = data

        # Store metadata
        self.metadata[data_id] = {
            'source_type': source_type,
            'ingestion_time': datetime.now(),
            'row_count': len(data),
            'columns': list(data.columns)
        }

        return True

    def validate_data_quality(self, data_id: str) -> DataQualityMetrics:
        """
        Validate data quality.

        Args:
            data_id: Data identifier

        Returns:
            Data quality metrics
        """
        if data_id not in self.data_cache:
            raise ValueError(f"Data {data_id} not found")

        data = self.data_cache[data_id]

        # Calculate metrics
        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        accuracy = 95.0  # Placeholder (would need reference data)
        timeliness = 90.0  # Placeholder (would need timestamp analysis)
        consistency = 92.0  # Placeholder (would need validation rules)

        overall = (completeness + accuracy + timeliness + consistency) / 4

        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            timeliness=timeliness,
            consistency=consistency,
            overall_quality=overall
        )

    def export_data(
        self,
        data_id: str,
        format: str = "csv"
    ) -> str:
        """
        Export data to file.

        Args:
            data_id: Data identifier
            format: Export format (csv, json, excel)

        Returns:
            Export file path
        """
        if data_id not in self.data_cache:
            raise ValueError(f"Data {data_id} not found")

        data = self.data_cache[data_id]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{data_id}_{timestamp}.{format}"

        if format == "csv":
            data.to_csv(filename, index=False)
        elif format == "json":
            data.to_json(filename, orient='records')
        elif format == "excel":
            data.to_excel(filename, index=False)

        return filename

    def get_data_summary(self, data_id: str) -> Dict[str, Any]:
        """Get data summary statistics."""
        if data_id not in self.data_cache:
            raise ValueError(f"Data {data_id} not found")

        data = self.data_cache[data_id]
        return {
            'row_count': len(data),
            'column_count': len(data.columns),
            'columns': list(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'metadata': self.metadata.get(data_id, {})
        }


# ============================================================================
# B15: MAIN APPLICATION INTEGRATION LAYER
# ============================================================================

class IntegrationStatus(str, Enum):
    """Integration status."""
    INITIALIZED = "Initialized"
    READY = "Ready"
    ERROR = "Error"
    PROCESSING = "Processing"


class ApplicationConfig(BaseModel):
    """Main application configuration."""

    app_name: str = Field(default="PV Circularity Simulator", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="production", description="Environment (dev/staging/production)")
    enable_design_suite: bool = Field(default=True, description="Enable design suite")
    enable_analysis_suite: bool = Field(default=True, description="Enable analysis suite")
    enable_monitoring_suite: bool = Field(default=True, description="Enable monitoring suite")
    enable_circularity_suite: bool = Field(default=True, description="Enable circularity suite")
    enable_financial_analysis: bool = Field(default=True, description="Enable financial analysis")
    debug_mode: bool = Field(default=False, description="Debug mode")


class ApplicationIntegrator:
    """
    Main Application Integration Layer.
    Orchestrates all suite modules and provides unified interface.
    """

    def __init__(self, config: Optional[ApplicationConfig] = None):
        """Initialize application integrator."""
        self.config = config or ApplicationConfig()
        self.status = IntegrationStatus.INITIALIZED
        self.data_manager = DataManager()
        self.financial_analyzer = FinancialAnalyzer()

        # Initialize suite connectors
        self.suite_modules: Dict[str, Any] = {}

    def initialize_suites(self) -> bool:
        """
        Initialize all enabled suite modules.

        Returns:
            Success status
        """
        try:
            if self.config.enable_design_suite:
                from . import design_suite
                self.suite_modules['design'] = design_suite.DesignSuite()

            if self.config.enable_analysis_suite:
                from . import analysis_suite
                self.suite_modules['analysis'] = analysis_suite.AnalysisSuite()

            if self.config.enable_monitoring_suite:
                from . import monitoring_suite
                self.suite_modules['monitoring'] = monitoring_suite.MonitoringSuite()

            if self.config.enable_circularity_suite:
                from . import circularity_suite
                self.suite_modules['circularity'] = circularity_suite.CircularitySuite()

            self.status = IntegrationStatus.READY
            return True

        except Exception as e:
            self.status = IntegrationStatus.ERROR
            print(f"Error initializing suites: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            'application': self.config.app_name,
            'version': self.config.version,
            'status': self.status,
            'enabled_modules': {
                'design_suite': self.config.enable_design_suite,
                'analysis_suite': self.config.enable_analysis_suite,
                'monitoring_suite': self.config.enable_monitoring_suite,
                'circularity_suite': self.config.enable_circularity_suite,
                'financial_analysis': self.config.enable_financial_analysis
            },
            'initialized_suites': list(self.suite_modules.keys()),
            'timestamp': datetime.now().isoformat()
        }

    def run_complete_analysis(
        self,
        pv_capacity_kw: float,
        location: Dict[str, float],
        financial_params: Optional[FinancialParameters] = None
    ) -> Dict[str, Any]:
        """
        Run complete integrated analysis across all suites.

        Args:
            pv_capacity_kw: PV system capacity
            location: Geographic location
            financial_params: Financial parameters

        Returns:
            Integrated analysis results
        """
        results = {
            'timestamp': datetime.now(),
            'system_capacity_kw': pv_capacity_kw,
            'location': location
        }

        # Design Suite Analysis
        if 'design' in self.suite_modules:
            design_results = self.suite_modules['design'].design_workflow(
                material_id="MAT001",
                cell_params=None,  # Would be populated with actual parameters
                module_config=None  # Would be populated with actual configuration
            )
            results['design'] = design_results

        # Analysis Suite
        if 'analysis' in self.suite_modules:
            analysis_results = self.suite_modules['analysis'].complete_system_analysis(
                module_power_wp=400,
                capacity_kw=pv_capacity_kw,
                location=location
            )
            results['analysis'] = analysis_results

        # Monitoring Suite
        if 'monitoring' in self.suite_modules:
            monitoring_results = self.suite_modules['monitoring'].monitor_and_diagnose()
            results['monitoring'] = monitoring_results

        # Circularity Suite
        if 'circularity' in self.suite_modules:
            circularity_results = self.suite_modules['circularity'].complete_circularity_analysis(
                system_age_years=5,
                system_capacity_kw=pv_capacity_kw,
                current_pr=85,
                module_efficiency=20,
                original_efficiency=21
            )
            results['circularity'] = circularity_results

        # Financial Analysis
        if financial_params:
            financial_results = self.financial_analyzer.analyze_project(financial_params)
            results['financial'] = financial_results.dict()

        return results


# ============================================================================
# APPLICATION SUITE INTEGRATION INTERFACE
# ============================================================================

class ApplicationSuite:
    """
    Unified Application Suite Interface integrating B13-B15.
    Provides complete application framework.
    """

    def __init__(self, config: Optional[ApplicationConfig] = None):
        """Initialize application suite."""
        self.config = config or ApplicationConfig()
        self.integrator = ApplicationIntegrator(self.config)
        self.data_manager = DataManager()
        self.financial_analyzer = FinancialAnalyzer()

    def start_application(self) -> bool:
        """Start the application."""
        success = self.integrator.initialize_suites()
        if success:
            print(f"✓ {self.config.app_name} v{self.config.version} started successfully")
        else:
            print(f"✗ Failed to start {self.config.app_name}")
        return success

    def get_status(self) -> Dict[str, Any]:
        """Get application status."""
        return self.integrator.get_system_status()


# Export main interface
__all__ = [
    'ApplicationSuite',
    'ApplicationIntegrator',
    'FinancialAnalyzer',
    'FinancialParameters',
    'FinancialResults',
    'DataManager',
    'ApplicationConfig'
]
