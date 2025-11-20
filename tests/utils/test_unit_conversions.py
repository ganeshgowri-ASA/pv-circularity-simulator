"""
Tests for unit conversion utilities.
"""

import pytest
from src.pv_simulator.utils.unit_conversions import (
    convert_energy,
    convert_power,
    convert_area,
    convert_mass,
    convert_length,
    convert_temperature,
    convert_efficiency,
    calculate_energy_from_power,
    calculate_power_from_energy,
    calculate_specific_yield,
)


class TestEnergyConversion:
    """Tests for energy conversion functions."""

    def test_wh_to_kwh(self):
        """Test Wh to kWh conversion."""
        assert convert_energy(1000, "Wh", "kWh") == 1.0
        assert convert_energy(5000, "Wh", "kWh") == 5.0

    def test_kwh_to_mwh(self):
        """Test kWh to MWh conversion."""
        assert convert_energy(1000, "kWh", "MWh") == 1.0

    def test_wh_to_joules(self):
        """Test Wh to J conversion."""
        result = convert_energy(1, "Wh", "J")
        assert abs(result - 3600) < 0.1

    def test_kwh_to_mj(self):
        """Test kWh to MJ conversion."""
        result = convert_energy(1, "kWh", "MJ")
        assert abs(result - 3.6) < 0.01

    def test_same_unit_conversion(self):
        """Test conversion to same unit."""
        assert convert_energy(100, "kWh", "kWh") == 100.0

    def test_invalid_unit(self):
        """Test invalid unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported energy unit"):
            convert_energy(100, "invalid", "kWh")


class TestPowerConversion:
    """Tests for power conversion functions."""

    def test_w_to_kw(self):
        """Test W to kW conversion."""
        assert convert_power(1000, "W", "kW") == 1.0
        assert convert_power(5500, "W", "kW") == 5.5

    def test_mw_to_kw(self):
        """Test MW to kW conversion."""
        assert convert_power(2, "MW", "kW") == 2000.0

    def test_kw_to_w(self):
        """Test kW to W conversion."""
        assert convert_power(0.5, "kW", "W") == 500.0

    def test_invalid_power_unit(self):
        """Test invalid power unit raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported power unit"):
            convert_power(100, "hp", "kW")


class TestAreaConversion:
    """Tests for area conversion functions."""

    def test_cm2_to_m2(self):
        """Test cm² to m² conversion."""
        assert convert_area(10000, "cm2", "m2") == 1.0

    def test_mm2_to_m2(self):
        """Test mm² to m² conversion."""
        assert convert_area(1000000, "mm2", "m2") == 1.0

    def test_hectare_to_m2(self):
        """Test hectare to m² conversion."""
        assert convert_area(1, "ha", "m2") == 10000.0

    def test_m2_to_km2(self):
        """Test m² to km² conversion."""
        assert convert_area(1000000, "m2", "km2") == 1.0


class TestMassConversion:
    """Tests for mass conversion functions."""

    def test_g_to_kg(self):
        """Test g to kg conversion."""
        assert convert_mass(1000, "g", "kg") == 1.0

    def test_ton_to_kg(self):
        """Test ton to kg conversion."""
        assert convert_mass(1, "ton", "kg") == 1000.0

    def test_lb_to_kg(self):
        """Test lb to kg conversion."""
        result = convert_mass(1, "lb", "kg")
        assert abs(result - 0.453592) < 0.0001

    def test_kg_to_g(self):
        """Test kg to g conversion."""
        assert convert_mass(2.5, "kg", "g") == 2500.0


class TestLengthConversion:
    """Tests for length conversion functions."""

    def test_cm_to_m(self):
        """Test cm to m conversion."""
        assert convert_length(100, "cm", "m") == 1.0

    def test_km_to_m(self):
        """Test km to m conversion."""
        assert convert_length(1, "km", "m") == 1000.0

    def test_inch_to_cm(self):
        """Test inch to cm conversion."""
        result = convert_length(1, "inch", "cm")
        assert abs(result - 2.54) < 0.01

    def test_ft_to_m(self):
        """Test foot to m conversion."""
        result = convert_length(1, "ft", "m")
        assert abs(result - 0.3048) < 0.0001


class TestTemperatureConversion:
    """Tests for temperature conversion functions."""

    def test_celsius_to_fahrenheit(self):
        """Test C to F conversion."""
        assert convert_temperature(0, "C", "F") == 32.0
        assert convert_temperature(100, "C", "F") == 212.0

    def test_fahrenheit_to_celsius(self):
        """Test F to C conversion."""
        assert convert_temperature(32, "F", "C") == 0.0
        result = convert_temperature(212, "F", "C")
        assert abs(result - 100) < 0.01

    def test_celsius_to_kelvin(self):
        """Test C to K conversion."""
        assert convert_temperature(0, "C", "K") == 273.15
        result = convert_temperature(25, "C", "K")
        assert abs(result - 298.15) < 0.01

    def test_kelvin_to_celsius(self):
        """Test K to C conversion."""
        assert convert_temperature(273.15, "K", "C") == 0.0

    def test_same_temperature(self):
        """Test conversion to same unit."""
        assert convert_temperature(25, "C", "C") == 25.0


class TestEfficiencyConversion:
    """Tests for efficiency conversion functions."""

    def test_percent_to_decimal(self):
        """Test percent to decimal conversion."""
        assert convert_efficiency(20, "percent", "decimal") == 0.2
        assert convert_efficiency(85.5, "percent", "decimal") == 0.855

    def test_decimal_to_percent(self):
        """Test decimal to percent conversion."""
        assert convert_efficiency(0.15, "decimal", "percent") == 15.0
        assert convert_efficiency(0.925, "decimal", "percent") == 92.5

    def test_ppm_to_percent(self):
        """Test ppm to percent conversion."""
        assert convert_efficiency(10000, "ppm", "percent") == 1.0


class TestEnergyPowerCalculations:
    """Tests for energy-power calculation functions."""

    def test_energy_from_power(self):
        """Test energy calculation from power."""
        # 1000W for 5 hours = 5000Wh
        assert calculate_energy_from_power(1000, "W", 5) == 5000.0

        # 2kW for 10 hours = 20000Wh
        assert calculate_energy_from_power(2, "kW", 10) == 20000.0

    def test_power_from_energy(self):
        """Test power calculation from energy."""
        # 5000Wh over 5 hours = 1000W
        assert calculate_power_from_energy(5000, "Wh", 5) == 1000.0

        # 20kWh over 10 hours = 2000W
        assert calculate_power_from_energy(20, "kWh", 10) == 2000.0

    def test_power_from_energy_zero_duration(self):
        """Test power calculation with zero duration raises error."""
        with pytest.raises(ValueError, match="Duration cannot be zero"):
            calculate_power_from_energy(1000, "Wh", 0)


class TestSpecificYield:
    """Tests for specific yield calculations."""

    def test_specific_yield_calculation(self):
        """Test specific yield calculation."""
        # 1000 kWh from 5 kW system = 200 kWh/kWp
        result = calculate_specific_yield(1000, "kWh", 5, "kW")
        assert abs(result - 200.0) < 0.1

        # 5000 Wh from 1000 W system = 5 kWh/kWp
        result = calculate_specific_yield(5000, "Wh", 1000, "W")
        assert abs(result - 5.0) < 0.1

    def test_specific_yield_zero_power(self):
        """Test specific yield with zero power raises error."""
        with pytest.raises(ValueError, match="Power cannot be zero"):
            calculate_specific_yield(1000, "kWh", 0, "kW")
