"""
Recycling Economics & Material Value (B11-S04)

This module provides tools for analyzing the economics of PV module recycling,
including material pricing, ROI calculations, and environmental credit valuation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime


class MaterialPrices(BaseModel):
    """Market prices for recovered materials (USD per kg)."""

    aluminum: float = Field(ge=0, description="Aluminum price USD/kg")
    glass: float = Field(ge=0, description="Glass cullet price USD/kg")
    silicon: float = Field(ge=0, description="Silicon price USD/kg")
    silver: float = Field(ge=0, description="Silver price USD/kg")
    copper: float = Field(ge=0, description="Copper price USD/kg")
    plastic: float = Field(default=0, ge=0, description="Plastic price USD/kg")

    @classmethod
    def current_market_prices(cls) -> "MaterialPrices":
        """Get current market prices (2024 estimates)."""
        return cls(
            aluminum=2.50,
            glass=0.05,
            silicon=15.0,
            silver=650.0,
            copper=8.50,
            plastic=0.30
        )


class RecyclingRevenue(BaseModel):
    """Revenue breakdown from recycled materials."""

    aluminum_revenue: float = Field(ge=0, description="Revenue from aluminum")
    glass_revenue: float = Field(ge=0, description="Revenue from glass")
    silicon_revenue: float = Field(ge=0, description="Revenue from silicon")
    silver_revenue: float = Field(ge=0, description="Revenue from silver")
    copper_revenue: float = Field(ge=0, description="Revenue from copper")
    other_revenue: float = Field(default=0, ge=0, description="Revenue from other materials")

    @property
    def total_revenue(self) -> float:
        """Total revenue from all materials."""
        return (
            self.aluminum_revenue + self.glass_revenue + self.silicon_revenue +
            self.silver_revenue + self.copper_revenue + self.other_revenue
        )


class EnvironmentalCredit(BaseModel):
    """Environmental credits and benefits."""

    carbon_credits_usd: float = Field(ge=0, description="Carbon credit value in USD")
    recycling_certificates_usd: float = Field(ge=0, description="Recycling certificate value")
    epr_compliance_value_usd: float = Field(ge=0, description="EPR compliance value")
    carbon_offset_tons: float = Field(ge=0, description="Carbon offset in tons CO2eq")
    landfill_avoidance_benefit_usd: float = Field(ge=0, description="Landfill avoidance benefit")

    @property
    def total_environmental_value(self) -> float:
        """Total environmental credit value."""
        return (
            self.carbon_credits_usd + self.recycling_certificates_usd +
            self.epr_compliance_value_usd + self.landfill_avoidance_benefit_usd
        )


class RecyclingROIAnalysis(BaseModel):
    """ROI analysis for recycling operations."""

    total_revenue: float = Field(description="Total revenue from recycling")
    total_costs: float = Field(ge=0, description="Total recycling costs")
    environmental_credits: float = Field(ge=0, description="Environmental credit value")
    net_profit: float = Field(description="Net profit (revenue + credits - costs)")
    roi_percent: float = Field(description="Return on investment percentage")
    payback_period_years: Optional[float] = Field(
        default=None,
        description="Payback period in years"
    )
    break_even_volume_modules: Optional[int] = Field(
        default=None,
        description="Break-even volume in modules per year"
    )


class RecyclingFacilityEconomics(BaseModel):
    """Economic analysis for recycling facility."""

    capacity_modules_per_year: int = Field(ge=0, description="Annual processing capacity")
    capital_investment: float = Field(ge=0, description="Initial capital investment")
    annual_operating_cost: float = Field(ge=0, description="Annual operating costs")
    annual_revenue: float = Field(ge=0, description="Annual revenue")
    annual_profit: float = Field(description="Annual profit/loss")
    facility_roi_percent: float = Field(description="Facility ROI percentage")
    npv_20_years: float = Field(description="20-year net present value")


class RecyclingEconomics:
    """
    Economic analysis tools for PV module recycling.

    Provides methods for material pricing, ROI calculations, and
    environmental credit valuation.
    """

    def __init__(
        self,
        material_prices: Optional[MaterialPrices] = None,
        carbon_price_per_ton: float = 50.0,
        discount_rate: float = 0.08
    ):
        """
        Initialize recycling economics analyzer.

        Args:
            material_prices: Market prices for materials
            carbon_price_per_ton: Carbon credit price in USD per ton CO2eq
            discount_rate: Discount rate for NPV calculations
        """
        self.material_prices = material_prices or MaterialPrices.current_market_prices()
        self.carbon_price = carbon_price_per_ton
        self.discount_rate = discount_rate

    def material_pricing(
        self,
        recovered_materials: Dict[str, float],
        material_quality: Dict[str, float] = None,
        market_conditions: str = "normal"
    ) -> RecyclingRevenue:
        """
        Calculate revenue from recovered materials based on current market prices.

        Args:
            recovered_materials: Dictionary of recovered material masses (kg)
            material_quality: Quality factors (0-1) affecting prices
            market_conditions: Market conditions ('depressed', 'normal', 'strong')

        Returns:
            RecyclingRevenue with detailed revenue breakdown
        """
        # Market condition multipliers
        market_multipliers = {
            "depressed": 0.75,
            "normal": 1.0,
            "strong": 1.25
        }
        multiplier = market_multipliers.get(market_conditions, 1.0)

        # Default quality factors
        if material_quality is None:
            material_quality = {
                "aluminum": 0.95,
                "glass": 0.90,
                "silicon": 0.85,
                "silver": 0.90,
                "copper": 0.95
            }

        # Calculate revenue for each material
        aluminum_rev = (
            recovered_materials.get("aluminum", 0) *
            self.material_prices.aluminum *
            material_quality.get("aluminum", 1.0) *
            multiplier
        )

        glass_rev = (
            recovered_materials.get("glass", 0) *
            self.material_prices.glass *
            material_quality.get("glass", 1.0) *
            multiplier
        )

        silicon_rev = (
            recovered_materials.get("silicon", 0) *
            self.material_prices.silicon *
            material_quality.get("silicon", 1.0) *
            multiplier
        )

        silver_rev = (
            recovered_materials.get("silver", 0) *
            self.material_prices.silver *
            material_quality.get("silver", 1.0) *
            multiplier
        )

        copper_rev = (
            recovered_materials.get("copper", 0) *
            self.material_prices.copper *
            material_quality.get("copper", 1.0) *
            multiplier
        )

        other_rev = (
            recovered_materials.get("other", 0) *
            self.material_prices.plastic *
            multiplier
        )

        return RecyclingRevenue(
            aluminum_revenue=aluminum_rev,
            glass_revenue=glass_rev,
            silicon_revenue=silicon_rev,
            silver_revenue=silver_rev,
            copper_revenue=copper_rev,
            other_revenue=other_rev
        )

    def recycling_roi(
        self,
        num_modules: int,
        avg_module_weight_kg: float,
        recycling_cost_per_module: float,
        recovered_materials: Dict[str, float],
        facility_investment: Optional[float] = None,
        annual_volume: Optional[int] = None
    ) -> RecyclingROIAnalysis:
        """
        Calculate ROI for recycling operations.

        Args:
            num_modules: Number of modules to recycle
            avg_module_weight_kg: Average module weight in kg
            recycling_cost_per_module: Cost per module for recycling
            recovered_materials: Recovered materials per module (kg)
            facility_investment: Initial facility investment (for payback calc)
            annual_volume: Annual recycling volume (for break-even calc)

        Returns:
            RecyclingROIAnalysis with detailed ROI metrics
        """
        # Calculate total costs
        total_costs = recycling_cost_per_module * num_modules

        # Calculate revenue from materials
        # Scale recovered materials by number of modules
        scaled_materials = {
            k: v * num_modules for k, v in recovered_materials.items()
        }
        revenue = self.material_pricing(scaled_materials)

        # Calculate environmental credits
        env_credits = self.environmental_credits(
            num_modules=num_modules,
            avg_module_weight_kg=avg_module_weight_kg
        )

        # Calculate net profit
        total_revenue = revenue.total_revenue
        total_env_value = env_credits.total_environmental_value
        net_profit = total_revenue + total_env_value - total_costs

        # Calculate ROI
        roi_percent = (net_profit / total_costs * 100) if total_costs > 0 else 0

        # Calculate payback period if investment provided
        payback_period = None
        if facility_investment and net_profit > 0 and annual_volume:
            annual_profit = net_profit * (annual_volume / num_modules)
            payback_period = facility_investment / annual_profit if annual_profit > 0 else None

        # Calculate break-even volume
        break_even_volume = None
        if recycling_cost_per_module > 0:
            revenue_per_module = total_revenue / num_modules if num_modules > 0 else 0
            credits_per_module = total_env_value / num_modules if num_modules > 0 else 0
            net_per_module = revenue_per_module + credits_per_module - recycling_cost_per_module

            if net_per_module > 0 and facility_investment:
                break_even_volume = int(np.ceil(facility_investment / net_per_module))

        return RecyclingROIAnalysis(
            total_revenue=total_revenue,
            total_costs=total_costs,
            environmental_credits=total_env_value,
            net_profit=net_profit,
            roi_percent=roi_percent,
            payback_period_years=payback_period,
            break_even_volume_modules=break_even_volume
        )

    def environmental_credits(
        self,
        num_modules: int,
        avg_module_weight_kg: float,
        region: str = "EU"
    ) -> EnvironmentalCredit:
        """
        Calculate environmental credits and benefits from recycling.

        Args:
            num_modules: Number of modules recycled
            avg_module_weight_kg: Average module weight in kg
            region: Region for regulatory framework ('EU', 'US', 'China', 'Other')

        Returns:
            EnvironmentalCredit with detailed environmental value
        """
        total_weight_kg = num_modules * avg_module_weight_kg
        total_weight_tons = total_weight_kg / 1000

        # Carbon offset calculation
        # Recycling vs virgin material production saves ~2.5 tons CO2eq per ton of material
        carbon_offset_tons = total_weight_tons * 2.5
        carbon_credits_usd = carbon_offset_tons * self.carbon_price

        # Recycling certificates (varies by region)
        certificate_values = {
            "EU": 0.10,  # EUR per kg
            "US": 0.05,
            "China": 0.03,
            "Other": 0.02
        }
        certificate_value_per_kg = certificate_values.get(region, 0.02)
        recycling_certificates = total_weight_kg * certificate_value_per_kg

        # EPR (Extended Producer Responsibility) compliance value
        # Manufacturers pay for recycling compliance
        epr_values = {
            "EU": 0.15,  # EUR per kg
            "US": 0.08,
            "China": 0.05,
            "Other": 0.03
        }
        epr_value_per_kg = epr_values.get(region, 0.03)
        epr_compliance = total_weight_kg * epr_value_per_kg

        # Landfill avoidance benefit
        # Cost of landfilling avoided
        landfill_cost_per_ton = 50.0
        landfill_avoidance = total_weight_tons * landfill_cost_per_ton

        return EnvironmentalCredit(
            carbon_credits_usd=carbon_credits_usd,
            recycling_certificates_usd=recycling_certificates,
            epr_compliance_value_usd=epr_compliance,
            carbon_offset_tons=carbon_offset_tons,
            landfill_avoidance_benefit_usd=landfill_avoidance
        )

    def facility_economics(
        self,
        capacity_modules_per_year: int,
        capital_investment: float,
        operating_cost_per_module: float,
        avg_module_weight_kg: float,
        recovery_rates: Dict[str, float],
        utilization_rate: float = 0.80,
        analysis_years: int = 20
    ) -> RecyclingFacilityEconomics:
        """
        Analyze economics of a recycling facility.

        Args:
            capacity_modules_per_year: Annual processing capacity
            capital_investment: Initial capital investment
            operating_cost_per_module: Operating cost per module
            avg_module_weight_kg: Average module weight
            recovery_rates: Material recovery rates (kg per module)
            utilization_rate: Facility utilization rate (0-1)
            analysis_years: Years for NPV analysis

        Returns:
            RecyclingFacilityEconomics with detailed facility analysis
        """
        # Actual annual volume
        actual_annual_volume = int(capacity_modules_per_year * utilization_rate)

        # Annual operating costs
        annual_operating_cost = operating_cost_per_module * actual_annual_volume

        # Annual revenue
        roi_analysis = self.recycling_roi(
            num_modules=actual_annual_volume,
            avg_module_weight_kg=avg_module_weight_kg,
            recycling_cost_per_module=operating_cost_per_module,
            recovered_materials=recovery_rates
        )
        annual_revenue = roi_analysis.total_revenue + roi_analysis.environmental_credits

        # Annual profit
        annual_profit = annual_revenue - annual_operating_cost

        # Facility ROI
        facility_roi = (annual_profit / capital_investment * 100) if capital_investment > 0 else 0

        # Calculate 20-year NPV
        npv = self._calculate_facility_npv(
            initial_investment=capital_investment,
            annual_cash_flow=annual_profit,
            years=analysis_years,
            discount_rate=self.discount_rate
        )

        return RecyclingFacilityEconomics(
            capacity_modules_per_year=capacity_modules_per_year,
            capital_investment=capital_investment,
            annual_operating_cost=annual_operating_cost,
            annual_revenue=annual_revenue,
            annual_profit=annual_profit,
            facility_roi_percent=facility_roi,
            npv_20_years=npv
        )

    def sensitivity_analysis(
        self,
        base_case: RecyclingROIAnalysis,
        num_modules: int,
        recovered_materials: Dict[str, float],
        recycling_cost_per_module: float,
        price_variations: List[float] = None
    ) -> Dict[str, List[Dict]]:
        """
        Perform sensitivity analysis on key economic variables.

        Args:
            base_case: Base case ROI analysis
            num_modules: Number of modules
            recovered_materials: Recovered materials per module
            recycling_cost_per_module: Cost per module
            price_variations: List of price variation factors (default: [0.5, 0.75, 1.0, 1.25, 1.5])

        Returns:
            Dictionary with sensitivity analysis results
        """
        if price_variations is None:
            price_variations = [0.5, 0.75, 1.0, 1.25, 1.5]

        results = {
            "material_prices": [],
            "recycling_costs": [],
            "carbon_prices": []
        }

        # Material price sensitivity
        original_prices = self.material_prices.model_copy()
        for factor in price_variations:
            # Adjust all material prices
            self.material_prices = MaterialPrices(
                aluminum=original_prices.aluminum * factor,
                glass=original_prices.glass * factor,
                silicon=original_prices.silicon * factor,
                silver=original_prices.silver * factor,
                copper=original_prices.copper * factor,
                plastic=original_prices.plastic * factor
            )

            roi = self.recycling_roi(
                num_modules=num_modules,
                avg_module_weight_kg=20.0,
                recycling_cost_per_module=recycling_cost_per_module,
                recovered_materials=recovered_materials
            )

            results["material_prices"].append({
                "price_factor": factor,
                "roi_percent": roi.roi_percent,
                "net_profit": roi.net_profit
            })

        # Restore original prices
        self.material_prices = original_prices

        # Recycling cost sensitivity
        for factor in price_variations:
            adjusted_cost = recycling_cost_per_module * factor
            roi = self.recycling_roi(
                num_modules=num_modules,
                avg_module_weight_kg=20.0,
                recycling_cost_per_module=adjusted_cost,
                recovered_materials=recovered_materials
            )

            results["recycling_costs"].append({
                "cost_factor": factor,
                "roi_percent": roi.roi_percent,
                "net_profit": roi.net_profit
            })

        # Carbon price sensitivity
        original_carbon_price = self.carbon_price
        for factor in price_variations:
            self.carbon_price = original_carbon_price * factor
            roi = self.recycling_roi(
                num_modules=num_modules,
                avg_module_weight_kg=20.0,
                recycling_cost_per_module=recycling_cost_per_module,
                recovered_materials=recovered_materials
            )

            results["carbon_prices"].append({
                "carbon_price_factor": factor,
                "roi_percent": roi.roi_percent,
                "net_profit": roi.net_profit
            })

        # Restore original carbon price
        self.carbon_price = original_carbon_price

        return results

    @staticmethod
    def _calculate_facility_npv(
        initial_investment: float,
        annual_cash_flow: float,
        years: int,
        discount_rate: float
    ) -> float:
        """Calculate net present value for facility investment."""
        if discount_rate == 0:
            return annual_cash_flow * years - initial_investment

        # NPV formula
        npv = -initial_investment
        for year in range(1, years + 1):
            npv += annual_cash_flow / ((1 + discount_rate) ** year)

        return npv

    def compare_scenarios(
        self,
        scenarios: List[Dict]
    ) -> List[Dict]:
        """
        Compare multiple recycling scenarios.

        Args:
            scenarios: List of scenario dictionaries with parameters

        Returns:
            List of scenario results with ROI analysis
        """
        results = []

        for i, scenario in enumerate(scenarios):
            roi = self.recycling_roi(
                num_modules=scenario.get("num_modules", 1000),
                avg_module_weight_kg=scenario.get("avg_module_weight_kg", 20.0),
                recycling_cost_per_module=scenario.get("recycling_cost_per_module", 15.0),
                recovered_materials=scenario.get("recovered_materials", {}),
                facility_investment=scenario.get("facility_investment"),
                annual_volume=scenario.get("annual_volume")
            )

            results.append({
                "scenario_name": scenario.get("name", f"Scenario {i+1}"),
                "parameters": scenario,
                "roi_analysis": roi.model_dump()
            })

        # Sort by ROI
        results.sort(key=lambda x: x["roi_analysis"]["roi_percent"], reverse=True)

        return results
