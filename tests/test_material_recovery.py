"""Tests for MaterialRecoveryCalculator."""

import pytest
from src.circularity.material_recovery import (
    MaterialRecoveryCalculator,
    ModuleComposition,
    RecoveryRates,
    RecyclingCostStructure,
)


@pytest.fixture
def standard_module():
    """Standard PV module composition."""
    return ModuleComposition(
        glass=15.0,
        aluminum=2.5,
        silicon=0.5,
        silver=0.005,
        copper=0.2,
        eva_polymer=1.0,
        backsheet=0.5,
        junction_box=0.3,
        other=0.0
    )


@pytest.fixture
def calculator():
    """Material recovery calculator instance."""
    return MaterialRecoveryCalculator()


def test_module_composition_total_mass(standard_module):
    """Test module total mass calculation."""
    expected_mass = 15.0 + 2.5 + 0.5 + 0.005 + 0.2 + 1.0 + 0.5 + 0.3
    assert abs(standard_module.total_mass - expected_mass) < 0.001


def test_metal_recovery(calculator, standard_module):
    """Test metal recovery calculation."""
    result = calculator.metal_recovery(standard_module, recovery_method="combined")

    assert "aluminum_kg" in result
    assert "silver_kg" in result
    assert "copper_kg" in result
    assert result["aluminum_kg"] <= standard_module.aluminum
    assert result["silver_kg"] <= standard_module.silver
    assert result["copper_kg"] <= standard_module.copper
    assert result["total_metal_kg"] > 0


def test_glass_recovery(calculator, standard_module):
    """Test glass recovery calculation."""
    result = calculator.glass_recovery(standard_module, processing_quality="standard")

    assert "recovered_glass_kg" in result
    assert result["recovered_glass_kg"] <= standard_module.glass
    assert 0 <= result["glass_recovery_rate"] <= 1
    assert 0 <= result["glass_purity"] <= 1


def test_silicon_recovery(calculator, standard_module):
    """Test silicon recovery calculation."""
    result = calculator.silicon_recovery(standard_module, recovery_technique="thermal_chemical")

    assert "recovered_silicon_kg" in result
    assert result["recovered_silicon_kg"] <= standard_module.silicon
    assert 0 <= result["silicon_recovery_rate"] <= 1
    assert result["solar_grade_silicon_kg"] + result["metallurgical_grade_silicon_kg"] == pytest.approx(
        result["recovered_silicon_kg"]
    )


def test_recycling_costs(calculator, standard_module):
    """Test recycling cost calculation."""
    result = calculator.recycling_costs(
        composition=standard_module,
        num_modules=100,
        transport_distance_km=100.0,
        process_method="combined"
    )

    assert result.total_cost > 0
    assert result.collection_cost >= 0
    assert result.transportation_cost >= 0
    assert result.dismantling_cost >= 0


def test_full_recovery_analysis(calculator, standard_module):
    """Test full recovery analysis."""
    result = calculator.full_recovery_analysis(
        composition=standard_module,
        num_modules=50,
        transport_distance_km=200.0
    )

    assert "metal_recovery" in result
    assert "glass_recovery" in result
    assert "silicon_recovery" in result
    assert "total_recovered_materials" in result
    assert "cost_breakdown" in result
    assert result["overall_recovery_rate"] > 0
    assert result["total_cost_usd"] > 0


def test_recovery_rates_validation():
    """Test recovery rates validation."""
    # Valid rates
    rates = RecoveryRates(
        glass=0.95,
        aluminum=0.98,
        silicon=0.85,
        silver=0.90,
        copper=0.95
    )
    assert rates.glass == 0.95

    # Invalid rates (should raise validation error)
    with pytest.raises(Exception):
        RecoveryRates(glass=1.5)  # > 1.0

    with pytest.raises(Exception):
        RecoveryRates(aluminum=-0.1)  # < 0


def test_different_recovery_methods(calculator, standard_module):
    """Test different recovery methods."""
    methods = ["thermal", "chemical", "mechanical", "combined"]

    for method in methods:
        result = calculator.metal_recovery(standard_module, recovery_method=method)
        assert result["total_metal_kg"] > 0
