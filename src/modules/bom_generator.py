"""BOM Generator and Cost Calculator for PV Module Manufacturing.

This module provides comprehensive Bill of Materials (BOM) generation and cost
calculation capabilities for photovoltaic module manufacturing, including:
- BOM generation from module designs
- Material cost calculations with pricing tiers
- Multi-currency support
- Supplier comparison and optimization
- Budget analysis
- Export functionality (CSV, Excel, JSON)
"""

import json
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import ValidationError

from src.modules.models import (
    BOMItem,
    BudgetAnalysis,
    ComponentCategory,
    CostBreakdown,
    Currency,
    CurrencyExchangeRate,
    Material,
    MaterialWithTiers,
    ModuleDesign,
    SupplierComparison,
)


class BOMGeneratorError(Exception):
    """Base exception for BOM Generator errors."""

    pass


class MaterialNotFoundError(BOMGeneratorError):
    """Raised when a required material is not found in the database."""

    pass


class InvalidModuleDesignError(BOMGeneratorError):
    """Raised when module design is invalid or incomplete."""

    pass


class ExportError(BOMGeneratorError):
    """Raised when BOM export fails."""

    pass


class BOMGenerator:
    """Bill of Materials Generator and Cost Calculator for PV Modules.

    This class provides comprehensive BOM generation and cost calculation
    functionality for photovoltaic module manufacturing.

    Attributes:
        materials_db: Dictionary of available materials keyed by material ID
        exchange_rates: Dictionary of currency exchange rates
        manufacturing_overhead_rate: Manufacturing overhead percentage (default: 15%)
        default_currency: Default currency for calculations

    Example:
        >>> from src.modules.bom_generator import BOMGenerator
        >>> from src.modules.models import Material, ModuleDesign
        >>>
        >>> # Initialize generator with materials database
        >>> materials = [Material(...), ...]
        >>> generator = BOMGenerator(materials)
        >>>
        >>> # Generate BOM from module design
        >>> module_design = ModuleDesign(...)
        >>> bom_df = generator.generate_bom(module_design)
        >>>
        >>> # Calculate costs
        >>> costs = generator.calculate_material_costs(bom_df, materials)
        >>> total_cost = generator.calculate_module_cost(costs)
        >>>
        >>> # Export BOM
        >>> csv_path = generator.export_bom(bom_df, format="csv")
    """

    def __init__(
        self,
        materials: Optional[List[Union[Material, MaterialWithTiers]]] = None,
        manufacturing_overhead_rate: float = 0.15,
        default_currency: Currency = Currency.USD,
    ) -> None:
        """Initialize BOM Generator with materials database.

        Args:
            materials: List of Material objects for the materials database
            manufacturing_overhead_rate: Manufacturing overhead as decimal (default: 0.15 = 15%)
            default_currency: Default currency for cost calculations

        Raises:
            ValueError: If manufacturing_overhead_rate is negative
        """
        if manufacturing_overhead_rate < 0:
            raise ValueError("Manufacturing overhead rate cannot be negative")

        self.materials_db: Dict[str, Union[Material, MaterialWithTiers]] = {}
        if materials:
            self.materials_db = {mat.id: mat for mat in materials}

        self.exchange_rates: Dict[tuple[Currency, Currency], CurrencyExchangeRate] = {}
        self.manufacturing_overhead_rate = manufacturing_overhead_rate
        self.default_currency = default_currency

    def add_material(self, material: Union[Material, MaterialWithTiers]) -> None:
        """Add a material to the materials database.

        Args:
            material: Material object to add
        """
        self.materials_db[material.id] = material

    def add_exchange_rate(self, exchange_rate: CurrencyExchangeRate) -> None:
        """Add or update currency exchange rate.

        Args:
            exchange_rate: CurrencyExchangeRate object
        """
        key = (exchange_rate.from_currency, exchange_rate.to_currency)
        self.exchange_rates[key] = exchange_rate

    def convert_currency(
        self, amount: Decimal, from_currency: Currency, to_currency: Currency
    ) -> Decimal:
        """Convert amount from one currency to another.

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Converted amount in target currency

        Raises:
            ValueError: If exchange rate is not available
        """
        if from_currency == to_currency:
            return amount

        key = (from_currency, to_currency)
        if key not in self.exchange_rates:
            raise ValueError(
                f"Exchange rate not available for {from_currency} to {to_currency}"
            )

        rate = self.exchange_rates[key].rate
        return amount * rate

    def generate_bom(self, module_design: Union[Dict, ModuleDesign]) -> pd.DataFrame:
        """Generate Bill of Materials from module design specification.

        This method creates a comprehensive BOM including all components required
        for manufacturing a PV module based on the design specification.

        Args:
            module_design: Module design specification (dict or ModuleDesign object)

        Returns:
            DataFrame containing BOM with columns: material_id, component_name,
            category, quantity, unit, supplier, notes

        Raises:
            InvalidModuleDesignError: If module design is invalid
            ValidationError: If module design validation fails

        Example:
            >>> design = {
            ...     "module_id": "MOD-001",
            ...     "num_cells": 60,
            ...     "cell_size": 156.75,
            ...     # ... other design parameters
            ... }
            >>> bom = generator.generate_bom(design)
        """
        # Validate and convert module design
        if isinstance(module_design, dict):
            try:
                design = ModuleDesign(**module_design)
            except ValidationError as e:
                raise InvalidModuleDesignError(f"Invalid module design: {e}") from e
        else:
            design = module_design

        bom_items: List[Dict] = []

        # Cell components
        num_cells = design.num_cells
        cell_area_m2 = (design.cell_size / 1000) ** 2  # Convert mm to m

        # Silicon wafer (per cell)
        bom_items.append(
            {
                "material_id": "MAT-WAFER-001",
                "component_name": "Silicon Wafer",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells)),
                "unit": "pieces",
                "supplier": "",
                "notes": f"Size: {design.cell_size}mm",
            }
        )

        # Emitter layer (per cell area)
        bom_items.append(
            {
                "material_id": "MAT-EMITTER-001",
                "component_name": "Emitter Layer (Phosphorus)",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells * cell_area_m2)),
                "unit": "m2",
                "supplier": "",
                "notes": "n-type emitter layer",
            }
        )

        # BSF (Back Surface Field) layer
        bom_items.append(
            {
                "material_id": "MAT-BSF-001",
                "component_name": "BSF Layer (Aluminum)",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells * cell_area_m2)),
                "unit": "m2",
                "supplier": "",
                "notes": "p+ back surface field",
            }
        )

        # Anti-reflective coating
        bom_items.append(
            {
                "material_id": "MAT-ARC-001",
                "component_name": "Anti-Reflective Coating",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells * cell_area_m2)),
                "unit": "m2",
                "supplier": "",
                "notes": "Silicon nitride coating",
            }
        )

        # Metallization - Front contacts
        bom_items.append(
            {
                "material_id": "MAT-METAL-FRONT-001",
                "component_name": "Front Metallization (Silver)",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells * 0.1)),  # ~0.1g silver per cell
                "unit": "g",
                "supplier": "",
                "notes": "Front contacts and busbars",
            }
        )

        # Metallization - Back contacts
        bom_items.append(
            {
                "material_id": "MAT-METAL-BACK-001",
                "component_name": "Back Metallization (Aluminum)",
                "category": ComponentCategory.CELL.value,
                "quantity": Decimal(str(num_cells * cell_area_m2 * 1000)),  # g/m2
                "unit": "g",
                "supplier": "",
                "notes": "Back contact layer",
            }
        )

        # Module components
        module_length = design.dimensions.get("length", 1640)  # mm
        module_width = design.dimensions.get("width", 990)  # mm
        module_area_m2 = (module_length * module_width) / 1_000_000  # Convert to m2

        # Front glass
        bom_items.append(
            {
                "material_id": "MAT-GLASS-001",
                "component_name": f"Front Glass ({design.glass_type})",
                "category": ComponentCategory.MODULE.value,
                "quantity": Decimal(str(module_area_m2)),
                "unit": "m2",
                "supplier": "",
                "notes": "3.2mm tempered glass",
            }
        )

        # Encapsulant (top and bottom layers)
        bom_items.append(
            {
                "material_id": "MAT-ENCAP-001",
                "component_name": f"Encapsulant ({design.encapsulant_type})",
                "category": ComponentCategory.MODULE.value,
                "quantity": Decimal(str(module_area_m2 * 2)),  # Top and bottom layers
                "unit": "m2",
                "supplier": "",
                "notes": "EVA or POE sheets",
            }
        )

        # Backsheet
        bom_items.append(
            {
                "material_id": "MAT-BACKSHEET-001",
                "component_name": f"Backsheet ({design.backsheet_type})",
                "category": ComponentCategory.MODULE.value,
                "quantity": Decimal(str(module_area_m2)),
                "unit": "m2",
                "supplier": "",
                "notes": "Weatherproof backsheet",
            }
        )

        # Frame
        frame_perimeter = 2 * (module_length + module_width) / 1000  # meters
        bom_items.append(
            {
                "material_id": "MAT-FRAME-001",
                "component_name": f"Frame ({design.frame_type})",
                "category": ComponentCategory.MODULE.value,
                "quantity": Decimal(str(frame_perimeter)),
                "unit": "m",
                "supplier": "",
                "notes": "Anodized aluminum frame",
            }
        )

        # Junction box
        bom_items.append(
            {
                "material_id": "MAT-JBOX-001",
                "component_name": f"Junction Box ({design.junction_box_type})",
                "category": ComponentCategory.MODULE.value,
                "quantity": Decimal("1"),
                "unit": "pieces",
                "supplier": "",
                "notes": "With bypass diodes",
            }
        )

        # Interconnect materials
        # Ribbon for cell interconnection
        ribbon_length_per_cell = design.cell_size * 2 / 1000  # meters (front + back)
        total_ribbon_length = ribbon_length_per_cell * num_cells

        bom_items.append(
            {
                "material_id": "MAT-RIBBON-001",
                "component_name": "Tabbing Ribbon (Copper)",
                "category": ComponentCategory.INTERCONNECT.value,
                "quantity": Decimal(str(total_ribbon_length)),
                "unit": "m",
                "supplier": "",
                "notes": "Tinned copper ribbon",
            }
        )

        # Bus ribbon
        bom_items.append(
            {
                "material_id": "MAT-BUSBAR-001",
                "component_name": "Bus Ribbon",
                "category": ComponentCategory.INTERCONNECT.value,
                "quantity": Decimal(str(module_length / 1000 * 5)),  # 5 bus ribbons
                "unit": "m",
                "supplier": "",
                "notes": "For string connections",
            }
        )

        # Solder
        bom_items.append(
            {
                "material_id": "MAT-SOLDER-001",
                "component_name": "Solder (SnPb or lead-free)",
                "category": ComponentCategory.INTERCONNECT.value,
                "quantity": Decimal(str(num_cells * 0.5)),  # ~0.5g per cell
                "unit": "g",
                "supplier": "",
                "notes": "Cell interconnection",
            }
        )

        # Flux
        bom_items.append(
            {
                "material_id": "MAT-FLUX-001",
                "component_name": "Soldering Flux",
                "category": ComponentCategory.INTERCONNECT.value,
                "quantity": Decimal(str(num_cells * 0.2)),  # ~0.2g per cell
                "unit": "g",
                "supplier": "",
                "notes": "For clean soldering",
            }
        )

        # Adhesives and sealants
        bom_items.append(
            {
                "material_id": "MAT-ADHESIVE-001",
                "component_name": "Edge Sealant",
                "category": ComponentCategory.ADHESIVE.value,
                "quantity": Decimal(str(frame_perimeter * 0.01)),  # kg
                "unit": "kg",
                "supplier": "",
                "notes": "Silicone edge seal",
            }
        )

        bom_items.append(
            {
                "material_id": "MAT-ADHESIVE-002",
                "component_name": "Junction Box Adhesive",
                "category": ComponentCategory.ADHESIVE.value,
                "quantity": Decimal("0.05"),  # kg
                "unit": "kg",
                "supplier": "",
                "notes": "Structural adhesive",
            }
        )

        # Convert to DataFrame
        bom_df = pd.DataFrame(bom_items)

        return bom_df

    def _get_material_price(
        self, material: Union[Material, MaterialWithTiers], quantity: Decimal
    ) -> Decimal:
        """Get material price considering volume-based pricing tiers.

        Args:
            material: Material object (with or without pricing tiers)
            quantity: Quantity to purchase

        Returns:
            Unit price for the given quantity
        """
        # Check if material has pricing tiers
        if isinstance(material, MaterialWithTiers) and material.pricing_tiers:
            # Find applicable pricing tier
            for tier in sorted(material.pricing_tiers, key=lambda t: t.min_quantity):
                if quantity >= tier.min_quantity:
                    if tier.max_quantity is None or quantity <= tier.max_quantity:
                        return tier.unit_price
            # If no tier matches, use base price
            return material.base_price
        else:
            # Use base price
            return material.base_price

    def calculate_material_costs(
        self,
        bom: pd.DataFrame,
        materials_db: Optional[List[Union[Material, MaterialWithTiers]]] = None,
    ) -> Dict:
        """Calculate material costs for BOM with waste factors and pricing tiers.

        Args:
            bom: BOM DataFrame from generate_bom()
            materials_db: Optional list of materials (uses self.materials_db if None)

        Returns:
            Dictionary containing:
                - bom_with_costs: DataFrame with cost columns added
                - cost_breakdown: CostBreakdown object with detailed costs
                - missing_materials: List of material IDs not found in database

        Raises:
            ValueError: If BOM DataFrame is empty or invalid

        Example:
            >>> costs = generator.calculate_material_costs(bom_df, materials)
            >>> print(f"Total cost: {costs['cost_breakdown'].total_cost}")
        """
        if bom.empty:
            raise ValueError("BOM DataFrame cannot be empty")

        # Use provided materials or instance materials
        if materials_db is not None:
            materials_dict = {mat.id: mat for mat in materials_db}
        else:
            materials_dict = self.materials_db

        # Track missing materials
        missing_materials: List[str] = []

        # Create a copy of BOM to add cost columns
        bom_with_costs = bom.copy()

        # Initialize cost columns
        bom_with_costs["unit_cost"] = Decimal("0")
        bom_with_costs["waste_adjusted_quantity"] = Decimal("0")
        bom_with_costs["total_cost"] = Decimal("0")
        bom_with_costs["transportation_cost"] = Decimal("0")

        # Calculate costs for each item
        for idx, row in bom_with_costs.iterrows():
            material_id = row["material_id"]

            if material_id not in materials_dict:
                missing_materials.append(material_id)
                continue

            material = materials_dict[material_id]
            quantity = Decimal(str(row["quantity"]))

            # Apply waste factor
            waste_adjusted_qty = quantity * (Decimal("1") + Decimal(str(material.waste_factor)))
            bom_with_costs.at[idx, "waste_adjusted_quantity"] = waste_adjusted_qty

            # Get unit price (considering volume tiers)
            unit_price = self._get_material_price(material, waste_adjusted_qty)

            # Convert currency if needed
            if material.currency != self.default_currency:
                try:
                    unit_price = self.convert_currency(
                        unit_price, material.currency, self.default_currency
                    )
                except ValueError:
                    # If conversion not available, use base price
                    pass

            bom_with_costs.at[idx, "unit_cost"] = unit_price
            bom_with_costs.at[idx, "supplier"] = material.supplier

            # Calculate total cost
            material_cost = unit_price * waste_adjusted_qty
            transportation_cost = material.transportation_cost_per_unit * waste_adjusted_qty

            bom_with_costs.at[idx, "total_cost"] = material_cost
            bom_with_costs.at[idx, "transportation_cost"] = transportation_cost

        # Calculate cost breakdown by category
        cost_breakdown = CostBreakdown(currency=self.default_currency)

        for category in ComponentCategory:
            category_items = bom_with_costs[bom_with_costs["category"] == category.value]
            category_cost = category_items["total_cost"].sum()

            if category == ComponentCategory.CELL:
                cost_breakdown.cell_costs = Decimal(str(category_cost))
            elif category == ComponentCategory.MODULE:
                cost_breakdown.module_costs = Decimal(str(category_cost))
            elif category == ComponentCategory.INTERCONNECT:
                cost_breakdown.interconnect_costs = Decimal(str(category_cost))
            elif category == ComponentCategory.ADHESIVE:
                cost_breakdown.adhesive_costs = Decimal(str(category_cost))
            elif category == ComponentCategory.PACKAGING:
                cost_breakdown.packaging_costs = Decimal(str(category_cost))

        # Calculate totals
        cost_breakdown.material_subtotal = (
            cost_breakdown.cell_costs
            + cost_breakdown.module_costs
            + cost_breakdown.interconnect_costs
            + cost_breakdown.adhesive_costs
            + cost_breakdown.packaging_costs
        )

        # Calculate waste costs
        total_without_waste = sum(
            Decimal(str(row["unit_cost"])) * Decimal(str(row["quantity"]))
            for _, row in bom_with_costs.iterrows()
        )
        cost_breakdown.waste_costs = cost_breakdown.material_subtotal - total_without_waste

        # Transportation costs
        cost_breakdown.transportation_costs = Decimal(
            str(bom_with_costs["transportation_cost"].sum())
        )

        # Manufacturing overhead
        cost_breakdown.manufacturing_overhead = cost_breakdown.material_subtotal * Decimal(
            str(self.manufacturing_overhead_rate)
        )

        # Total cost
        cost_breakdown.total_cost = (
            cost_breakdown.material_subtotal
            + cost_breakdown.transportation_costs
            + cost_breakdown.manufacturing_overhead
        )

        return {
            "bom_with_costs": bom_with_costs,
            "cost_breakdown": cost_breakdown,
            "missing_materials": missing_materials,
        }

    def calculate_module_cost(self, bom_with_costs: Dict) -> float:
        """Calculate total module cost from cost breakdown.

        Args:
            bom_with_costs: Dictionary returned from calculate_material_costs()

        Returns:
            Total module cost as float

        Example:
            >>> costs = generator.calculate_material_costs(bom_df, materials)
            >>> total = generator.calculate_module_cost(costs)
        """
        cost_breakdown = bom_with_costs["cost_breakdown"]
        return float(cost_breakdown.total_cost)

    def export_bom(
        self, bom: pd.DataFrame, format: str = "csv", output_path: Optional[str] = None
    ) -> str:
        """Export BOM to various formats.

        Args:
            bom: BOM DataFrame to export
            format: Export format - 'csv', 'excel', or 'json'
            output_path: Optional output file path (auto-generated if None)

        Returns:
            Path to the exported file

        Raises:
            ExportError: If export fails
            ValueError: If format is not supported

        Example:
            >>> path = generator.export_bom(bom_df, format="excel")
            >>> print(f"BOM exported to {path}")
        """
        format = format.lower()
        if format not in ["csv", "excel", "json"]:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'")

        # Generate output path if not provided
        if output_path is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"bom_export_{timestamp}.{format if format != 'excel' else 'xlsx'}"

        try:
            output_file = Path(output_path)

            if format == "csv":
                bom.to_csv(output_file, index=False)
            elif format == "excel":
                bom.to_excel(output_file, index=False, engine="openpyxl")
            elif format == "json":
                # Convert Decimal to float for JSON serialization
                bom_json = bom.copy()
                for col in bom_json.select_dtypes(include=["object"]).columns:
                    bom_json[col] = bom_json[col].apply(
                        lambda x: float(x) if isinstance(x, Decimal) else x
                    )
                bom_json.to_json(output_file, orient="records", indent=2)

            return str(output_file.absolute())

        except Exception as e:
            raise ExportError(f"Failed to export BOM: {e}") from e

    def optimize_bom_cost(self, bom: pd.DataFrame) -> pd.DataFrame:
        """Optimize BOM costs by selecting best suppliers and materials.

        This method analyzes the BOM and suggests optimizations:
        - Selecting suppliers with lowest costs
        - Applying volume discounts where applicable
        - Recommending alternative materials if available

        Args:
            bom: BOM DataFrame with cost information

        Returns:
            Optimized BOM DataFrame with cost savings

        Example:
            >>> optimized_bom = generator.optimize_bom_cost(bom_with_costs_df)
        """
        optimized_bom = bom.copy()

        # Track optimization opportunities
        for idx, row in optimized_bom.iterrows():
            material_id = row["material_id"]
            quantity = Decimal(str(row.get("quantity", 0)))

            # Find alternative materials in same category
            category = row["category"]
            alternatives = [
                mat
                for mat in self.materials_db.values()
                if mat.category == category and mat.id != material_id
            ]

            if alternatives:
                current_material = self.materials_db.get(material_id)
                if current_material:
                    current_price = self._get_material_price(current_material, quantity)

                    # Find cheapest alternative
                    best_alternative = None
                    best_price = current_price

                    for alt_material in alternatives:
                        alt_price = self._get_material_price(alt_material, quantity)
                        if alt_price < best_price:
                            best_price = alt_price
                            best_alternative = alt_material

                    # If better alternative found, suggest it
                    if best_alternative:
                        savings = (current_price - best_price) * quantity
                        optimized_bom.at[idx, "material_id"] = best_alternative.id
                        optimized_bom.at[idx, "supplier"] = best_alternative.supplier
                        optimized_bom.at[idx, "unit_cost"] = best_price
                        optimized_bom.at[idx, "notes"] = (
                            f"Optimized from {material_id}. Savings: {float(savings):.2f}"
                        )

        return optimized_bom

    def compare_suppliers(self, material_name: str) -> SupplierComparison:
        """Compare suppliers for a specific material.

        Args:
            material_name: Name of the material to compare

        Returns:
            SupplierComparison object with supplier pricing and recommendations

        Raises:
            MaterialNotFoundError: If no materials found with given name

        Example:
            >>> comparison = generator.compare_suppliers("Silicon Wafer")
            >>> print(f"Recommended: {comparison.recommended_supplier}")
        """
        # Find all materials with matching name
        matching_materials = [
            mat for mat in self.materials_db.values() if material_name.lower() in mat.name.lower()
        ]

        if not matching_materials:
            raise MaterialNotFoundError(f"No materials found with name: {material_name}")

        # Build supplier comparison
        supplier_prices: Dict[str, Decimal] = {}

        for material in matching_materials:
            supplier = material.supplier
            price = material.base_price

            # Convert to default currency if needed
            if material.currency != self.default_currency:
                try:
                    price = self.convert_currency(
                        price, material.currency, self.default_currency
                    )
                except ValueError:
                    pass

            supplier_prices[supplier] = price

        # Find recommended supplier (lowest price)
        recommended_supplier = min(supplier_prices, key=supplier_prices.get)  # type: ignore
        lowest_price = supplier_prices[recommended_supplier]
        highest_price = max(supplier_prices.values())

        potential_savings = highest_price - lowest_price

        return SupplierComparison(
            material_name=material_name,
            suppliers=supplier_prices,
            recommended_supplier=recommended_supplier,
            potential_savings=potential_savings,
        )

    def analyze_budget(self, budgeted_cost: float, actual_cost: float) -> BudgetAnalysis:
        """Analyze budget vs actual costs.

        Args:
            budgeted_cost: Budgeted cost
            actual_cost: Actual cost

        Returns:
            BudgetAnalysis object with variance analysis

        Example:
            >>> analysis = generator.analyze_budget(10000.0, 9500.0)
            >>> print(f"Under budget by {analysis.variance_percentage}%")
        """
        budgeted = Decimal(str(budgeted_cost))
        actual = Decimal(str(actual_cost))

        variance = actual - budgeted
        variance_pct = float((variance / budgeted) * Decimal("100")) if budgeted > 0 else 0.0
        over_budget = actual > budgeted

        return BudgetAnalysis(
            budgeted_cost=budgeted,
            actual_cost=actual,
            variance=variance,
            variance_percentage=variance_pct,
            over_budget=over_budget,
        )

    def get_cost_breakdown_by_category(self, bom_with_costs: pd.DataFrame) -> Dict[str, float]:
        """Get cost breakdown by component category.

        Args:
            bom_with_costs: BOM DataFrame with cost information

        Returns:
            Dictionary mapping category names to total costs

        Example:
            >>> breakdown = generator.get_cost_breakdown_by_category(bom_df)
            >>> for category, cost in breakdown.items():
            ...     print(f"{category}: ${cost:.2f}")
        """
        breakdown: Dict[str, float] = {}

        for category in ComponentCategory:
            category_items = bom_with_costs[bom_with_costs["category"] == category.value]
            total_cost = float(category_items["total_cost"].sum())
            breakdown[category.value] = total_cost

        return breakdown
