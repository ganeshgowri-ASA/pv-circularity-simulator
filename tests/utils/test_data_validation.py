"""
Tests for data validation utilities.
"""

import pytest
from datetime import datetime, date
from pydantic import ValidationError

from src.pv_simulator.utils.data_validation import (
    validate_positive,
    validate_non_negative,
    validate_percentage,
    validate_efficiency,
    validate_range,
    validate_list_not_empty,
    validate_dict_keys,
    validate_email,
    validate_date_range,
    safe_validate,
    batch_validate,
    PVModuleSpecs,
    EnergyProductionData,
    MaterialComposition,
)


class TestBasicValidators:
    """Tests for basic validation functions."""

    def test_validate_positive(self):
        """Test positive value validation."""
        assert validate_positive(5.0) == 5.0
        assert validate_positive(0.1) == 0.1
        assert validate_positive(100) == 100.0

    def test_validate_positive_zero(self):
        """Test zero raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0)

    def test_validate_positive_negative(self):
        """Test negative raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-5)

    def test_validate_non_negative(self):
        """Test non-negative value validation."""
        assert validate_non_negative(0) == 0.0
        assert validate_non_negative(5.5) == 5.5

    def test_validate_non_negative_negative(self):
        """Test negative raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_non_negative(-1)

    def test_validate_percentage(self):
        """Test percentage validation."""
        assert validate_percentage(0) == 0.0
        assert validate_percentage(50) == 50.0
        assert validate_percentage(100) == 100.0

    def test_validate_percentage_out_of_range(self):
        """Test out of range percentage raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            validate_percentage(101)

        with pytest.raises(ValueError, match="must be between 0 and 100"):
            validate_percentage(-1)

    def test_validate_efficiency(self):
        """Test efficiency validation."""
        assert validate_efficiency(0) == 0.0
        assert validate_efficiency(0.25) == 0.25
        assert validate_efficiency(1) == 1.0

    def test_validate_efficiency_out_of_range(self):
        """Test out of range efficiency raises error."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            validate_efficiency(1.5)


class TestRangeValidator:
    """Tests for range validation."""

    def test_validate_range_within(self):
        """Test value within range."""
        assert validate_range(5, 0, 10) == 5.0

    def test_validate_range_boundaries(self):
        """Test boundary values."""
        assert validate_range(0, 0, 10) == 0.0
        assert validate_range(10, 0, 10) == 10.0

    def test_validate_range_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ValueError, match="must be between"):
            validate_range(-1, 0, 10)

    def test_validate_range_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValueError, match="must be between"):
            validate_range(11, 0, 10)

    def test_validate_range_min_only(self):
        """Test with only minimum value."""
        assert validate_range(50, min_value=0) == 50.0

        with pytest.raises(ValueError, match="must be at least"):
            validate_range(-1, min_value=0)

    def test_validate_range_max_only(self):
        """Test with only maximum value."""
        assert validate_range(5, max_value=10) == 5.0

        with pytest.raises(ValueError, match="must be at most"):
            validate_range(15, max_value=10)


class TestListValidator:
    """Tests for list validation."""

    def test_validate_list_not_empty(self):
        """Test non-empty list validation."""
        test_list = [1, 2, 3]
        assert validate_list_not_empty(test_list) == test_list

    def test_validate_empty_list(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_list_not_empty([])


class TestDictValidator:
    """Tests for dictionary validation."""

    def test_validate_dict_keys_present(self):
        """Test all required keys present."""
        data = {"a": 1, "b": 2, "c": 3}
        required = ["a", "b"]
        assert validate_dict_keys(data, required) == data

    def test_validate_dict_keys_missing(self):
        """Test missing required keys."""
        data = {"a": 1}
        required = ["a", "b", "c"]

        with pytest.raises(ValueError, match="missing required keys"):
            validate_dict_keys(data, required)


class TestEmailValidator:
    """Tests for email validation."""

    def test_validate_valid_emails(self):
        """Test valid email addresses."""
        assert validate_email("user@example.com") == "user@example.com"
        assert validate_email("test.user@domain.co.uk") == "test.user@domain.co.uk"

    def test_validate_invalid_emails(self):
        """Test invalid email addresses."""
        with pytest.raises(ValueError, match="Invalid email format"):
            validate_email("invalid-email")

        with pytest.raises(ValueError, match="Invalid email format"):
            validate_email("@example.com")

        with pytest.raises(ValueError, match="Invalid email format"):
            validate_email("user@")


class TestDateRangeValidator:
    """Tests for date range validation."""

    def test_validate_valid_date_range(self):
        """Test valid date range."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        result = validate_date_range(start, end)
        assert result == (start, end)

    def test_validate_invalid_date_range(self):
        """Test invalid date range."""
        start = date(2024, 12, 31)
        end = date(2024, 1, 1)

        with pytest.raises(ValueError, match="end_date must be after start_date"):
            validate_date_range(start, end)


class TestSafeValidate:
    """Tests for safe validation."""

    def test_safe_validate_success(self):
        """Test successful validation."""
        result = safe_validate(validate_positive, 5)
        assert result == 5.0

    def test_safe_validate_failure_with_default(self):
        """Test failed validation returns default."""
        result = safe_validate(validate_positive, -1, default=0)
        assert result == 0

    def test_safe_validate_failure_without_default(self):
        """Test failed validation returns None."""
        result = safe_validate(validate_positive, -1)
        assert result is None


class TestBatchValidate:
    """Tests for batch validation."""

    def test_batch_validate_all_valid(self):
        """Test batch validation with all valid values."""
        values = [1, 2, 3, 4, 5]
        valid, errors = batch_validate(values, validate_positive)

        assert len(valid) == 5
        assert len(errors) == 0

    def test_batch_validate_some_invalid(self):
        """Test batch validation with some invalid values."""
        values = [1, -1, 5, -3, 10]
        valid, errors = batch_validate(values, validate_positive)

        assert len(valid) == 3
        assert valid == [1.0, 5.0, 10.0]
        assert len(errors) == 2
        assert errors[0][0] == 1  # Index of first error
        assert errors[1][0] == 3  # Index of second error


class TestPVModuleSpecs:
    """Tests for PV module specifications validator."""

    def test_valid_pv_module(self):
        """Test valid PV module specifications."""
        module = PVModuleSpecs(
            name="Test Module 300W",
            power_rating_w=300,
            efficiency=0.18,
            area_m2=1.67,
            voltage_voc=45.0,
            current_isc=9.5,
            temperature_coeff_power=-0.4,
            warranty_years=25,
        )

        assert module.name == "Test Module 300W"
        assert module.power_rating_w == 300

    def test_invalid_efficiency(self):
        """Test invalid efficiency raises error."""
        with pytest.raises(ValidationError):
            PVModuleSpecs(
                name="Test Module",
                power_rating_w=300,
                efficiency=1.5,  # Invalid: > 1
                area_m2=1.67,
                voltage_voc=45.0,
                current_isc=9.5,
                temperature_coeff_power=-0.4,
                warranty_years=25,
            )

    def test_empty_name(self):
        """Test empty name raises error."""
        with pytest.raises(ValidationError):
            PVModuleSpecs(
                name="",
                power_rating_w=300,
                efficiency=0.18,
                area_m2=1.67,
                voltage_voc=45.0,
                current_isc=9.5,
                temperature_coeff_power=-0.4,
                warranty_years=25,
            )


class TestEnergyProductionData:
    """Tests for energy production data validator."""

    def test_valid_energy_data(self):
        """Test valid energy production data."""
        data = EnergyProductionData(
            timestamp=datetime(2024, 1, 1, 12, 0),
            energy_kwh=5.5,
            power_kw=2.2,
            irradiance_w_m2=800,
            temperature_c=35,
            performance_ratio=0.85,
        )

        assert data.energy_kwh == 5.5
        assert data.power_kw == 2.2

    def test_negative_energy(self):
        """Test negative energy raises error."""
        with pytest.raises(ValidationError):
            EnergyProductionData(
                timestamp=datetime(2024, 1, 1, 12, 0),
                energy_kwh=-1,  # Invalid
                power_kw=2.2,
            )

    def test_irradiance_out_of_range(self):
        """Test irradiance out of range raises error."""
        with pytest.raises(ValidationError):
            EnergyProductionData(
                timestamp=datetime(2024, 1, 1, 12, 0),
                energy_kwh=5.5,
                power_kw=2.2,
                irradiance_w_m2=2000,  # Invalid: > 1500
            )


class TestMaterialComposition:
    """Tests for material composition validator."""

    def test_valid_material(self):
        """Test valid material composition."""
        material = MaterialComposition(
            material_name="Silicon",
            mass_kg=5.0,
            recyclability_rate=0.95,
            toxicity_level="low",
            cost_per_kg=50.0,
        )

        assert material.material_name == "Silicon"
        assert material.recyclability_rate == 0.95

    def test_invalid_recyclability(self):
        """Test invalid recyclability raises error."""
        with pytest.raises(ValidationError):
            MaterialComposition(
                material_name="Silicon",
                mass_kg=5.0,
                recyclability_rate=1.5,  # Invalid: > 1
            )

    def test_invalid_toxicity_level(self):
        """Test invalid toxicity level raises error."""
        with pytest.raises(ValidationError):
            MaterialComposition(
                material_name="Lead",
                mass_kg=0.5,
                recyclability_rate=0.8,
                toxicity_level="invalid",  # Invalid level
            )

    def test_valid_toxicity_levels(self):
        """Test all valid toxicity levels."""
        valid_levels = ["low", "medium", "high", "very_high"]

        for level in valid_levels:
            material = MaterialComposition(
                material_name="Test",
                mass_kg=1.0,
                recyclability_rate=0.5,
                toxicity_level=level,
            )
            assert material.toxicity_level == level
