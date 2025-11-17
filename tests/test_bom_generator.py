"""Comprehensive unit tests for BOM Generator and Cost Calculator."""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.modules.bom_generator import (
    BOMGenerator,
    BOMGeneratorError,
    ExportError,
    InvalidModuleDesignError,
    MaterialNotFoundError,
)
from src.modules.models import (
    ComponentCategory,
    Currency,
    CurrencyExchangeRate,
    Material,
    MaterialWithTiers,
    ModuleDesign,
    PricingTier,
)


@pytest.fixture
def sample_materials():
    """Create sample materials database for testing."""
    materials = [
        Material(
            id="MAT-WAFER-001",
            name="Silicon Wafer",
            category=ComponentCategory.CELL,
            supplier="Supplier A",
            unit="pieces",
            base_price=Decimal("5.50"),
            currency=Currency.USD,
            waste_factor=0.05,
            transportation_cost_per_unit=Decimal("0.10"),
        ),
        Material(
            id="MAT-EMITTER-001",
            name="Emitter Layer (Phosphorus)",
            category=ComponentCategory.CELL,
            supplier="Supplier B",
            unit="m2",
            base_price=Decimal("2.00"),
            currency=Currency.USD,
            waste_factor=0.03,
            transportation_cost_per_unit=Decimal("0.05"),
        ),
        Material(
            id="MAT-BSF-001",
            name="BSF Layer (Aluminum)",
            category=ComponentCategory.CELL,
            supplier="Supplier A",
            unit="m2",
            base_price=Decimal("1.50"),
            currency=Currency.USD,
            waste_factor=0.03,
        ),
        Material(
            id="MAT-ARC-001",
            name="Anti-Reflective Coating",
            category=ComponentCategory.CELL,
            supplier="Supplier C",
            unit="m2",
            base_price=Decimal("3.00"),
            currency=Currency.USD,
            waste_factor=0.02,
        ),
        Material(
            id="MAT-METAL-FRONT-001",
            name="Front Metallization (Silver)",
            category=ComponentCategory.CELL,
            supplier="Supplier D",
            unit="g",
            base_price=Decimal("0.80"),
            currency=Currency.USD,
            waste_factor=0.10,
        ),
        Material(
            id="MAT-METAL-BACK-001",
            name="Back Metallization (Aluminum)",
            category=ComponentCategory.CELL,
            supplier="Supplier A",
            unit="g",
            base_price=Decimal("0.05"),
            currency=Currency.USD,
            waste_factor=0.05,
        ),
        Material(
            id="MAT-GLASS-001",
            name="Front Glass (tempered)",
            category=ComponentCategory.MODULE,
            supplier="Supplier E",
            unit="m2",
            base_price=Decimal("15.00"),
            currency=Currency.USD,
            waste_factor=0.08,
            transportation_cost_per_unit=Decimal("1.00"),
        ),
        Material(
            id="MAT-ENCAP-001",
            name="Encapsulant (EVA)",
            category=ComponentCategory.MODULE,
            supplier="Supplier F",
            unit="m2",
            base_price=Decimal("8.00"),
            currency=Currency.USD,
            waste_factor=0.05,
        ),
        Material(
            id="MAT-BACKSHEET-001",
            name="Backsheet (standard)",
            category=ComponentCategory.MODULE,
            supplier="Supplier G",
            unit="m2",
            base_price=Decimal("12.00"),
            currency=Currency.USD,
            waste_factor=0.06,
        ),
        Material(
            id="MAT-FRAME-001",
            name="Frame (aluminum)",
            category=ComponentCategory.MODULE,
            supplier="Supplier H",
            unit="m",
            base_price=Decimal("3.50"),
            currency=Currency.USD,
            waste_factor=0.05,
            transportation_cost_per_unit=Decimal("0.20"),
        ),
        Material(
            id="MAT-JBOX-001",
            name="Junction Box (standard)",
            category=ComponentCategory.MODULE,
            supplier="Supplier I",
            unit="pieces",
            base_price=Decimal("8.00"),
            currency=Currency.USD,
            waste_factor=0.02,
        ),
        Material(
            id="MAT-RIBBON-001",
            name="Tabbing Ribbon (Copper)",
            category=ComponentCategory.INTERCONNECT,
            supplier="Supplier J",
            unit="m",
            base_price=Decimal("0.50"),
            currency=Currency.USD,
            waste_factor=0.10,
        ),
        Material(
            id="MAT-BUSBAR-001",
            name="Bus Ribbon",
            category=ComponentCategory.INTERCONNECT,
            supplier="Supplier J",
            unit="m",
            base_price=Decimal("0.75"),
            currency=Currency.USD,
            waste_factor=0.08,
        ),
        Material(
            id="MAT-SOLDER-001",
            name="Solder (SnPb or lead-free)",
            category=ComponentCategory.INTERCONNECT,
            supplier="Supplier K",
            unit="g",
            base_price=Decimal("0.10"),
            currency=Currency.USD,
            waste_factor=0.15,
        ),
        Material(
            id="MAT-FLUX-001",
            name="Soldering Flux",
            category=ComponentCategory.INTERCONNECT,
            supplier="Supplier K",
            unit="g",
            base_price=Decimal("0.05"),
            currency=Currency.USD,
            waste_factor=0.20,
        ),
        Material(
            id="MAT-ADHESIVE-001",
            name="Edge Sealant",
            category=ComponentCategory.ADHESIVE,
            supplier="Supplier L",
            unit="kg",
            base_price=Decimal("25.00"),
            currency=Currency.USD,
            waste_factor=0.10,
        ),
        Material(
            id="MAT-ADHESIVE-002",
            name="Junction Box Adhesive",
            category=ComponentCategory.ADHESIVE,
            supplier="Supplier L",
            unit="kg",
            base_price=Decimal("30.00"),
            currency=Currency.USD,
            waste_factor=0.05,
        ),
    ]
    return materials


@pytest.fixture
def sample_materials_with_tiers():
    """Create sample materials with pricing tiers."""
    materials = [
        MaterialWithTiers(
            id="MAT-WAFER-002",
            name="Silicon Wafer Premium",
            category=ComponentCategory.CELL,
            supplier="Supplier Z",
            unit="pieces",
            base_price=Decimal("6.00"),
            currency=Currency.USD,
            waste_factor=0.03,
            pricing_tiers=[
                PricingTier(
                    min_quantity=Decimal("0"),
                    max_quantity=Decimal("100"),
                    discount_percentage=0,
                    unit_price=Decimal("6.00"),
                ),
                PricingTier(
                    min_quantity=Decimal("100"),
                    max_quantity=Decimal("500"),
                    discount_percentage=10,
                    unit_price=Decimal("5.40"),
                ),
                PricingTier(
                    min_quantity=Decimal("500"),
                    max_quantity=None,
                    discount_percentage=20,
                    unit_price=Decimal("4.80"),
                ),
            ],
        ),
    ]
    return materials


@pytest.fixture
def sample_module_design():
    """Create sample module design for testing."""
    return ModuleDesign(
        module_id="MOD-TEST-001",
        module_type="mono-Si",
        power_rating=400.0,
        efficiency=20.5,
        dimensions={"length": 1640, "width": 990, "thickness": 35},
        num_cells=60,
        cell_size=156.75,
        frame_type="aluminum",
        glass_type="tempered",
        backsheet_type="standard",
        encapsulant_type="EVA",
        junction_box_type="standard",
    )


@pytest.fixture
def bom_generator(sample_materials):
    """Create BOM generator instance with sample materials."""
    return BOMGenerator(materials=sample_materials, manufacturing_overhead_rate=0.15)


@pytest.fixture
def exchange_rates():
    """Create sample exchange rates."""
    return [
        CurrencyExchangeRate(
            from_currency=Currency.USD,
            to_currency=Currency.EUR,
            rate=Decimal("0.92"),
        ),
        CurrencyExchangeRate(
            from_currency=Currency.EUR,
            to_currency=Currency.USD,
            rate=Decimal("1.09"),
        ),
        CurrencyExchangeRate(
            from_currency=Currency.USD,
            to_currency=Currency.CNY,
            rate=Decimal("7.25"),
        ),
    ]


class TestBOMGenerator:
    """Test BOM Generator initialization and basic functionality."""

    def test_initialization(self, sample_materials):
        """Test BOM generator initialization."""
        generator = BOMGenerator(materials=sample_materials)
        assert len(generator.materials_db) == len(sample_materials)
        assert generator.manufacturing_overhead_rate == 0.15
        assert generator.default_currency == Currency.USD

    def test_initialization_custom_overhead(self, sample_materials):
        """Test initialization with custom overhead rate."""
        generator = BOMGenerator(materials=sample_materials, manufacturing_overhead_rate=0.20)
        assert generator.manufacturing_overhead_rate == 0.20

    def test_initialization_negative_overhead(self, sample_materials):
        """Test initialization with negative overhead raises error."""
        with pytest.raises(ValueError, match="Manufacturing overhead rate cannot be negative"):
            BOMGenerator(materials=sample_materials, manufacturing_overhead_rate=-0.05)

    def test_initialization_empty_materials(self):
        """Test initialization with empty materials list."""
        generator = BOMGenerator(materials=[])
        assert len(generator.materials_db) == 0

    def test_add_material(self, bom_generator):
        """Test adding material to database."""
        initial_count = len(bom_generator.materials_db)
        new_material = Material(
            id="MAT-TEST-001",
            name="Test Material",
            category=ComponentCategory.MODULE,
            supplier="Test Supplier",
            unit="kg",
            base_price=Decimal("10.00"),
        )
        bom_generator.add_material(new_material)
        assert len(bom_generator.materials_db) == initial_count + 1
        assert "MAT-TEST-001" in bom_generator.materials_db


class TestCurrencyConversion:
    """Test currency conversion functionality."""

    def test_add_exchange_rate(self, bom_generator, exchange_rates):
        """Test adding exchange rates."""
        for rate in exchange_rates:
            bom_generator.add_exchange_rate(rate)
        assert len(bom_generator.exchange_rates) == len(exchange_rates)

    def test_convert_currency_same(self, bom_generator):
        """Test currency conversion with same currency."""
        amount = Decimal("100.00")
        result = bom_generator.convert_currency(amount, Currency.USD, Currency.USD)
        assert result == amount

    def test_convert_currency_different(self, bom_generator, exchange_rates):
        """Test currency conversion between different currencies."""
        for rate in exchange_rates:
            bom_generator.add_exchange_rate(rate)

        amount = Decimal("100.00")
        result = bom_generator.convert_currency(amount, Currency.USD, Currency.EUR)
        expected = amount * Decimal("0.92")
        assert result == expected

    def test_convert_currency_missing_rate(self, bom_generator):
        """Test currency conversion with missing exchange rate."""
        amount = Decimal("100.00")
        with pytest.raises(ValueError, match="Exchange rate not available"):
            bom_generator.convert_currency(amount, Currency.USD, Currency.JPY)


class TestBOMGeneration:
    """Test BOM generation from module designs."""

    def test_generate_bom_from_design(self, bom_generator, sample_module_design):
        """Test BOM generation from module design."""
        bom = bom_generator.generate_bom(sample_module_design)

        assert isinstance(bom, pd.DataFrame)
        assert not bom.empty
        assert "material_id" in bom.columns
        assert "component_name" in bom.columns
        assert "category" in bom.columns
        assert "quantity" in bom.columns
        assert "unit" in bom.columns

        # Check that all component categories are represented
        categories = set(bom["category"].unique())
        assert ComponentCategory.CELL.value in categories
        assert ComponentCategory.MODULE.value in categories
        assert ComponentCategory.INTERCONNECT.value in categories
        assert ComponentCategory.ADHESIVE.value in categories

    def test_generate_bom_from_dict(self, bom_generator):
        """Test BOM generation from dictionary."""
        design_dict = {
            "module_id": "MOD-TEST-002",
            "module_type": "poly-Si",
            "power_rating": 350.0,
            "efficiency": 18.5,
            "dimensions": {"length": 1640, "width": 990, "thickness": 35},
            "num_cells": 60,
            "cell_size": 156.75,
        }
        bom = bom_generator.generate_bom(design_dict)
        assert isinstance(bom, pd.DataFrame)
        assert not bom.empty

    def test_generate_bom_invalid_design(self, bom_generator):
        """Test BOM generation with invalid design."""
        invalid_design = {"module_id": "INVALID"}  # Missing required fields
        with pytest.raises(InvalidModuleDesignError):
            bom_generator.generate_bom(invalid_design)

    def test_generate_bom_cell_components(self, bom_generator, sample_module_design):
        """Test that BOM includes all cell components."""
        bom = bom_generator.generate_bom(sample_module_design)
        cell_components = bom[bom["category"] == ComponentCategory.CELL.value]

        component_names = cell_components["component_name"].tolist()
        assert "Silicon Wafer" in component_names
        assert "Emitter Layer (Phosphorus)" in component_names
        assert "BSF Layer (Aluminum)" in component_names
        assert "Anti-Reflective Coating" in component_names
        assert "Front Metallization (Silver)" in component_names
        assert "Back Metallization (Aluminum)" in component_names

    def test_generate_bom_module_components(self, bom_generator, sample_module_design):
        """Test that BOM includes all module components."""
        bom = bom_generator.generate_bom(sample_module_design)
        module_components = bom[bom["category"] == ComponentCategory.MODULE.value]

        component_names = module_components["component_name"].tolist()
        assert any("Front Glass" in name for name in component_names)
        assert any("Encapsulant" in name for name in component_names)
        assert any("Backsheet" in name for name in component_names)
        assert any("Frame" in name for name in component_names)
        assert any("Junction Box" in name for name in component_names)

    def test_generate_bom_quantities(self, bom_generator, sample_module_design):
        """Test that BOM quantities are calculated correctly."""
        bom = bom_generator.generate_bom(sample_module_design)

        # Check wafer quantity equals num_cells
        wafer_row = bom[bom["material_id"] == "MAT-WAFER-001"]
        assert not wafer_row.empty
        assert Decimal(str(wafer_row.iloc[0]["quantity"])) == Decimal(
            str(sample_module_design.num_cells)
        )


class TestCostCalculation:
    """Test cost calculation functionality."""

    def test_calculate_material_costs(self, bom_generator, sample_module_design):
        """Test material cost calculation."""
        bom = bom_generator.generate_bom(sample_module_design)
        result = bom_generator.calculate_material_costs(bom)

        assert "bom_with_costs" in result
        assert "cost_breakdown" in result
        assert "missing_materials" in result

        bom_with_costs = result["bom_with_costs"]
        assert "unit_cost" in bom_with_costs.columns
        assert "total_cost" in bom_with_costs.columns
        assert "waste_adjusted_quantity" in bom_with_costs.columns

    def test_calculate_material_costs_breakdown(self, bom_generator, sample_module_design):
        """Test cost breakdown calculation."""
        bom = bom_generator.generate_bom(sample_module_design)
        result = bom_generator.calculate_material_costs(bom)

        cost_breakdown = result["cost_breakdown"]
        assert cost_breakdown.cell_costs >= 0
        assert cost_breakdown.module_costs >= 0
        assert cost_breakdown.interconnect_costs >= 0
        assert cost_breakdown.adhesive_costs >= 0
        assert cost_breakdown.material_subtotal >= 0
        assert cost_breakdown.total_cost >= 0

    def test_calculate_material_costs_missing_materials(self, bom_generator, sample_module_design):
        """Test cost calculation with missing materials."""
        # Create generator with limited materials
        limited_generator = BOMGenerator(materials=[], manufacturing_overhead_rate=0.15)
        bom = bom_generator.generate_bom(sample_module_design)

        result = limited_generator.calculate_material_costs(bom)
        assert len(result["missing_materials"]) > 0

    def test_calculate_material_costs_empty_bom(self, bom_generator):
        """Test cost calculation with empty BOM."""
        empty_bom = pd.DataFrame()
        with pytest.raises(ValueError, match="BOM DataFrame cannot be empty"):
            bom_generator.calculate_material_costs(empty_bom)

    def test_calculate_module_cost(self, bom_generator, sample_module_design):
        """Test total module cost calculation."""
        bom = bom_generator.generate_bom(sample_module_design)
        result = bom_generator.calculate_material_costs(bom)
        total_cost = bom_generator.calculate_module_cost(result)

        assert isinstance(total_cost, float)
        assert total_cost > 0

    def test_pricing_tiers(self, sample_materials_with_tiers):
        """Test volume-based pricing tiers."""
        generator = BOMGenerator(materials=sample_materials_with_tiers)
        material = sample_materials_with_tiers[0]

        # Test tier 1 (0-100)
        price_low = generator._get_material_price(material, Decimal("50"))
        assert price_low == Decimal("6.00")

        # Test tier 2 (100-500)
        price_mid = generator._get_material_price(material, Decimal("200"))
        assert price_mid == Decimal("5.40")

        # Test tier 3 (500+)
        price_high = generator._get_material_price(material, Decimal("1000"))
        assert price_high == Decimal("4.80")

    def test_waste_factor_application(self, bom_generator, sample_module_design):
        """Test that waste factors are applied correctly."""
        bom = bom_generator.generate_bom(sample_module_design)
        result = bom_generator.calculate_material_costs(bom)

        bom_with_costs = result["bom_with_costs"]
        for _, row in bom_with_costs.iterrows():
            if row["material_id"] in bom_generator.materials_db:
                material = bom_generator.materials_db[row["material_id"]]
                quantity = Decimal(str(row["quantity"]))
                waste_adjusted = Decimal(str(row["waste_adjusted_quantity"]))
                expected = quantity * (Decimal("1") + Decimal(str(material.waste_factor)))
                assert waste_adjusted == expected


class TestBOMExport:
    """Test BOM export functionality."""

    def test_export_csv(self, bom_generator, sample_module_design, tmp_path):
        """Test CSV export."""
        bom = bom_generator.generate_bom(sample_module_design)
        output_path = tmp_path / "test_bom.csv"

        result_path = bom_generator.export_bom(bom, format="csv", output_path=str(output_path))
        assert Path(result_path).exists()
        assert result_path.endswith(".csv")

        # Verify content
        loaded_bom = pd.read_csv(result_path)
        assert len(loaded_bom) == len(bom)

    def test_export_excel(self, bom_generator, sample_module_design, tmp_path):
        """Test Excel export."""
        bom = bom_generator.generate_bom(sample_module_design)
        output_path = tmp_path / "test_bom.xlsx"

        result_path = bom_generator.export_bom(bom, format="excel", output_path=str(output_path))
        assert Path(result_path).exists()
        assert result_path.endswith(".xlsx")

        # Verify content
        loaded_bom = pd.read_excel(result_path, engine="openpyxl")
        assert len(loaded_bom) == len(bom)

    def test_export_json(self, bom_generator, sample_module_design, tmp_path):
        """Test JSON export."""
        bom = bom_generator.generate_bom(sample_module_design)
        output_path = tmp_path / "test_bom.json"

        result_path = bom_generator.export_bom(bom, format="json", output_path=str(output_path))
        assert Path(result_path).exists()
        assert result_path.endswith(".json")

        # Verify content
        with open(result_path) as f:
            loaded_data = json.load(f)
        assert len(loaded_data) == len(bom)

    def test_export_invalid_format(self, bom_generator, sample_module_design):
        """Test export with invalid format."""
        bom = bom_generator.generate_bom(sample_module_design)
        with pytest.raises(ValueError, match="Unsupported format"):
            bom_generator.export_bom(bom, format="xml")

    def test_export_auto_path(self, bom_generator, sample_module_design):
        """Test export with auto-generated path."""
        bom = bom_generator.generate_bom(sample_module_design)
        result_path = bom_generator.export_bom(bom, format="csv")

        assert Path(result_path).exists()
        assert "bom_export_" in result_path


class TestBOMOptimization:
    """Test BOM cost optimization functionality."""

    def test_optimize_bom_cost(self, sample_module_design):
        """Test BOM cost optimization."""
        # Create materials with different prices for same category
        materials = [
            Material(
                id="MAT-GLASS-001",
                name="Front Glass Option 1",
                category=ComponentCategory.MODULE,
                supplier="Supplier A",
                unit="m2",
                base_price=Decimal("20.00"),
                currency=Currency.USD,
            ),
            Material(
                id="MAT-GLASS-002",
                name="Front Glass Option 2",
                category=ComponentCategory.MODULE,
                supplier="Supplier B",
                unit="m2",
                base_price=Decimal("15.00"),  # Cheaper alternative
                currency=Currency.USD,
            ),
        ]

        generator = BOMGenerator(materials=materials)

        # Create simple BOM with expensive option
        bom = pd.DataFrame(
            [
                {
                    "material_id": "MAT-GLASS-001",
                    "component_name": "Front Glass",
                    "category": ComponentCategory.MODULE.value,
                    "quantity": Decimal("2.0"),
                    "unit": "m2",
                    "supplier": "Supplier A",
                    "notes": "",
                }
            ]
        )

        optimized_bom = generator.optimize_bom_cost(bom)

        # Should suggest cheaper alternative
        assert optimized_bom.iloc[0]["material_id"] == "MAT-GLASS-002"
        assert "Optimized" in optimized_bom.iloc[0]["notes"]


class TestSupplierComparison:
    """Test supplier comparison functionality."""

    def test_compare_suppliers(self):
        """Test supplier comparison."""
        materials = [
            Material(
                id="MAT-WAFER-S1",
                name="Silicon Wafer",
                category=ComponentCategory.CELL,
                supplier="Supplier A",
                unit="pieces",
                base_price=Decimal("6.00"),
                currency=Currency.USD,
            ),
            Material(
                id="MAT-WAFER-S2",
                name="Silicon Wafer",
                category=ComponentCategory.CELL,
                supplier="Supplier B",
                unit="pieces",
                base_price=Decimal("5.50"),
                currency=Currency.USD,
            ),
            Material(
                id="MAT-WAFER-S3",
                name="Silicon Wafer",
                category=ComponentCategory.CELL,
                supplier="Supplier C",
                unit="pieces",
                base_price=Decimal("6.25"),
                currency=Currency.USD,
            ),
        ]

        generator = BOMGenerator(materials=materials)
        comparison = generator.compare_suppliers("Silicon Wafer")

        assert comparison.material_name == "Silicon Wafer"
        assert len(comparison.suppliers) == 3
        assert comparison.recommended_supplier == "Supplier B"
        assert comparison.potential_savings == Decimal("0.75")

    def test_compare_suppliers_not_found(self, bom_generator):
        """Test supplier comparison with non-existent material."""
        with pytest.raises(MaterialNotFoundError):
            bom_generator.compare_suppliers("Non-existent Material")


class TestBudgetAnalysis:
    """Test budget analysis functionality."""

    def test_analyze_budget_under(self, bom_generator):
        """Test budget analysis when under budget."""
        analysis = bom_generator.analyze_budget(budgeted_cost=10000.0, actual_cost=9500.0)

        assert analysis.budgeted_cost == Decimal("10000.0")
        assert analysis.actual_cost == Decimal("9500.0")
        assert analysis.variance == Decimal("-500.0")
        assert analysis.variance_percentage == -5.0
        assert not analysis.over_budget

    def test_analyze_budget_over(self, bom_generator):
        """Test budget analysis when over budget."""
        analysis = bom_generator.analyze_budget(budgeted_cost=10000.0, actual_cost=10500.0)

        assert analysis.budgeted_cost == Decimal("10000.0")
        assert analysis.actual_cost == Decimal("10500.0")
        assert analysis.variance == Decimal("500.0")
        assert analysis.variance_percentage == 5.0
        assert analysis.over_budget

    def test_analyze_budget_exact(self, bom_generator):
        """Test budget analysis when exactly on budget."""
        analysis = bom_generator.analyze_budget(budgeted_cost=10000.0, actual_cost=10000.0)

        assert analysis.variance == Decimal("0.0")
        assert analysis.variance_percentage == 0.0
        assert not analysis.over_budget


class TestCostBreakdownByCategory:
    """Test cost breakdown by category."""

    def test_get_cost_breakdown_by_category(self, bom_generator, sample_module_design):
        """Test getting cost breakdown by category."""
        bom = bom_generator.generate_bom(sample_module_design)
        result = bom_generator.calculate_material_costs(bom)
        bom_with_costs = result["bom_with_costs"]

        breakdown = bom_generator.get_cost_breakdown_by_category(bom_with_costs)

        assert isinstance(breakdown, dict)
        assert ComponentCategory.CELL.value in breakdown
        assert ComponentCategory.MODULE.value in breakdown
        assert ComponentCategory.INTERCONNECT.value in breakdown
        assert ComponentCategory.ADHESIVE.value in breakdown

        # All costs should be non-negative
        for category, cost in breakdown.items():
            assert cost >= 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_cells_design(self, bom_generator):
        """Test with zero cells design."""
        invalid_design = {
            "module_id": "MOD-ZERO",
            "module_type": "mono-Si",
            "power_rating": 400.0,
            "efficiency": 20.5,
            "dimensions": {"length": 1640, "width": 990, "thickness": 35},
            "num_cells": 0,  # Invalid
            "cell_size": 156.75,
        }

        with pytest.raises(InvalidModuleDesignError):
            bom_generator.generate_bom(invalid_design)

    def test_negative_quantity(self, bom_generator):
        """Test with negative quantity in BOM."""
        invalid_bom = pd.DataFrame(
            [
                {
                    "material_id": "MAT-GLASS-001",
                    "component_name": "Front Glass",
                    "category": ComponentCategory.MODULE.value,
                    "quantity": -1.0,  # Invalid
                    "unit": "m2",
                    "supplier": "",
                    "notes": "",
                }
            ]
        )

        # Should handle gracefully or raise appropriate error
        result = bom_generator.calculate_material_costs(invalid_bom)
        assert "missing_materials" in result

    def test_very_large_module(self, bom_generator):
        """Test with very large module design."""
        large_design = {
            "module_id": "MOD-LARGE",
            "module_type": "mono-Si",
            "power_rating": 1000.0,
            "efficiency": 25.0,
            "dimensions": {"length": 3000, "width": 2000, "thickness": 40},
            "num_cells": 144,
            "cell_size": 210.0,
        }

        bom = bom_generator.generate_bom(large_design)
        assert not bom.empty
        assert len(bom) > 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, bom_generator, sample_module_design, tmp_path):
        """Test complete BOM generation and cost calculation workflow."""
        # Generate BOM
        bom = bom_generator.generate_bom(sample_module_design)
        assert not bom.empty

        # Calculate costs
        result = bom_generator.calculate_material_costs(bom)
        assert "bom_with_costs" in result
        assert "cost_breakdown" in result

        # Get total cost
        total_cost = bom_generator.calculate_module_cost(result)
        assert total_cost > 0

        # Analyze budget
        budget_analysis = bom_generator.analyze_budget(
            budgeted_cost=total_cost * 1.1, actual_cost=total_cost
        )
        assert not budget_analysis.over_budget

        # Export to all formats
        for format in ["csv", "excel", "json"]:
            output_path = tmp_path / f"bom.{format if format != 'excel' else 'xlsx'}"
            exported_path = bom_generator.export_bom(
                result["bom_with_costs"], format=format, output_path=str(output_path)
            )
            assert Path(exported_path).exists()

    def test_multi_currency_workflow(self, bom_generator, sample_module_design, exchange_rates):
        """Test workflow with multiple currencies."""
        # Add exchange rates
        for rate in exchange_rates:
            bom_generator.add_exchange_rate(rate)

        # Add material in different currency
        eur_material = Material(
            id="MAT-EUR-001",
            name="European Glass",
            category=ComponentCategory.MODULE,
            supplier="EU Supplier",
            unit="m2",
            base_price=Decimal("18.00"),
            currency=Currency.EUR,
        )
        bom_generator.add_material(eur_material)

        # Test currency conversion
        converted = bom_generator.convert_currency(
            Decimal("100.00"), Currency.USD, Currency.EUR
        )
        assert converted == Decimal("92.00")
