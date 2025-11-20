"""
Unit tests for GridInteraction class.

Tests grid code compliance, reactive power control, frequency regulation,
and smart grid communication functionality.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from pv_circularity_simulator.grid.grid_interaction import (
    ComplianceCheckResult,
    FrequencyRegulationConfig,
    GridCodeLimits,
    GridCodeStandard,
    GridConnectionState,
    GridInteraction,
    GridState,
    PowerQualityMetric,
    ReactivePowerControlConfig,
    SCADAConfig,
    SCADAProtocol,
)


class TestGridCodeLimits:
    """Test GridCodeLimits Pydantic model."""

    def test_valid_limits(self) -> None:
        """Test creation of valid grid code limits."""
        limits = GridCodeLimits(
            standard=GridCodeStandard.IEEE_1547,
            voltage_min=0.88,
            voltage_max=1.10,
            frequency_min=59.3,
            frequency_max=60.5,
        )
        assert limits.standard == GridCodeStandard.IEEE_1547
        assert limits.voltage_min == 0.88
        assert limits.voltage_max == 1.10

    def test_invalid_voltage_range(self) -> None:
        """Test that invalid voltage range raises error."""
        with pytest.raises(ValueError, match="voltage_min must be less than voltage_max"):
            GridCodeLimits(
                standard=GridCodeStandard.IEEE_1547,
                voltage_min=1.10,
                voltage_max=0.88,
                frequency_min=59.3,
                frequency_max=60.5,
            )

    def test_invalid_frequency_range(self) -> None:
        """Test that invalid frequency range raises error."""
        with pytest.raises(ValueError, match="frequency_min must be less than frequency_max"):
            GridCodeLimits(
                standard=GridCodeStandard.IEEE_1547,
                voltage_min=0.88,
                voltage_max=1.10,
                frequency_min=60.5,
                frequency_max=59.3,
            )

    def test_frozen_model(self) -> None:
        """Test that GridCodeLimits is immutable."""
        limits = GridCodeLimits(
            standard=GridCodeStandard.IEEE_1547,
            voltage_min=0.88,
            voltage_max=1.10,
            frequency_min=59.3,
            frequency_max=60.5,
        )
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            limits.voltage_min = 0.9  # type: ignore


class TestGridState:
    """Test GridState Pydantic model."""

    def test_valid_grid_state(self) -> None:
        """Test creation of valid grid state."""
        state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=50.0,
            active_power=2300.0,
            reactive_power=0.0,
        )
        assert state.voltage_l1 == 230.0
        assert state.frequency == 50.0

    def test_three_phase_state(self) -> None:
        """Test three-phase grid state."""
        state = GridState(
            voltage_l1=230.0,
            voltage_l2=231.0,
            voltage_l3=229.0,
            current_l1=10.0,
            current_l2=10.1,
            current_l3=9.9,
            frequency=50.0,
            active_power=6900.0,
            reactive_power=100.0,
        )
        # Test average voltage calculation
        expected_avg = (230.0 + 231.0 + 229.0) / 3
        assert abs(state.voltage_avg - expected_avg) < 0.01

    def test_apparent_power_calculation(self) -> None:
        """Test apparent power calculation."""
        state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=50.0,
            active_power=2000.0,
            reactive_power=1000.0,
        )
        expected_s = np.sqrt(2000.0**2 + 1000.0**2)
        assert abs(state.apparent_power - expected_s) < 0.01

    def test_invalid_frequency(self) -> None:
        """Test that invalid frequency raises error."""
        with pytest.raises(ValueError):
            GridState(
                voltage_l1=230.0,
                current_l1=10.0,
                frequency=100.0,  # Too high
                active_power=2300.0,
                reactive_power=0.0,
            )


class TestReactivePowerControlConfig:
    """Test ReactivePowerControlConfig Pydantic model."""

    def test_fixed_pf_mode(self) -> None:
        """Test fixed power factor mode configuration."""
        config = ReactivePowerControlConfig(
            mode="fixed_pf", target_power_factor=0.95
        )
        assert config.mode == "fixed_pf"
        assert config.target_power_factor == 0.95

    def test_volt_var_mode(self) -> None:
        """Test volt-var mode configuration."""
        config = ReactivePowerControlConfig(mode="volt_var")
        assert config.mode == "volt_var"
        # Check default volt-var curve
        assert config.volt_var_v1 == 0.92
        assert config.volt_var_v4 == 1.08

    def test_invalid_mode(self) -> None:
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError):
            ReactivePowerControlConfig(mode="invalid_mode")


class TestSCADAConfig:
    """Test SCADAConfig Pydantic model."""

    def test_valid_modbus_config(self) -> None:
        """Test valid Modbus configuration."""
        config = SCADAConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.100",
            port=502,
            device_id="INV001",
        )
        assert config.protocol == SCADAProtocol.MODBUS_TCP
        assert config.port == 502

    def test_invalid_port(self) -> None:
        """Test that invalid port raises error."""
        with pytest.raises(ValueError):
            SCADAConfig(
                protocol=SCADAProtocol.MODBUS_TCP,
                host="192.168.1.100",
                port=99999,  # Too high
                device_id="INV001",
            )


class TestGridInteraction:
    """Test GridInteraction class."""

    def test_initialization(self) -> None:
        """Test GridInteraction initialization."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )
        assert grid.grid_code_standard == GridCodeStandard.IEEE_1547
        assert grid.nominal_voltage == 230.0
        assert grid.rated_power == 5000.0

    def test_invalid_voltage(self) -> None:
        """Test that invalid nominal voltage raises error."""
        with pytest.raises(ValueError, match="nominal_voltage must be positive"):
            GridInteraction(
                grid_code_standard=GridCodeStandard.IEEE_1547,
                nominal_voltage=-230.0,
                rated_power=5000.0,
            )

    def test_invalid_rated_power(self) -> None:
        """Test that invalid rated power raises error."""
        with pytest.raises(ValueError, match="rated_power must be positive"):
            GridInteraction(
                grid_code_standard=GridCodeStandard.IEEE_1547,
                nominal_voltage=230.0,
                rated_power=-5000.0,
            )

    def test_load_grid_code_limits_ieee1547(self) -> None:
        """Test loading IEEE 1547 grid code limits."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )
        limits = grid.grid_code_limits
        assert limits.standard == GridCodeStandard.IEEE_1547
        assert limits.voltage_min == 0.88
        assert limits.voltage_max == 1.10
        assert limits.frequency_min == 59.3
        assert limits.frequency_max == 60.5

    def test_load_grid_code_limits_vde(self) -> None:
        """Test loading VDE-AR-N 4105 grid code limits."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.VDE_AR_N_4105,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )
        limits = grid.grid_code_limits
        assert limits.standard == GridCodeStandard.VDE_AR_N_4105
        assert limits.voltage_min == 0.90
        assert limits.voltage_max == 1.10
        assert limits.frequency_min == 47.5
        assert limits.frequency_max == 51.5


class TestGridCodeCompliance:
    """Test grid_code_compliance method."""

    def test_compliant_state(self) -> None:
        """Test that compliant grid state passes."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=230.0,  # 1.0 p.u.
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
            voltage_thd=2.0,
            current_thd=3.0,
            power_factor=1.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is True
        assert len(result.violations) == 0

    def test_voltage_violation_high(self) -> None:
        """Test voltage too high violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=260.0,  # 1.13 p.u. > 1.10 limit
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("above maximum" in v for v in result.violations)

    def test_voltage_violation_low(self) -> None:
        """Test voltage too low violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=195.0,  # 0.848 p.u. < 0.88 limit
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("below minimum" in v for v in result.violations)

    def test_frequency_violation_high(self) -> None:
        """Test frequency too high violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=61.0,  # Above 60.5 Hz limit
            active_power=2300.0,
            reactive_power=0.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("Frequency" in v and "above maximum" in v for v in result.violations)

    def test_frequency_violation_low(self) -> None:
        """Test frequency too low violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=59.0,  # Below 59.3 Hz limit
            active_power=2300.0,
            reactive_power=0.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("Frequency" in v and "below minimum" in v for v in result.violations)

    def test_thd_violation(self) -> None:
        """Test THD violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
            voltage_thd=8.0,  # Above 5% limit
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("THD" in v for v in result.violations)

    def test_power_factor_violation(self) -> None:
        """Test power factor violation."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
            power_factor=0.90,  # Below 0.95 limit
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert any("Power factor" in v for v in result.violations)

    def test_multiple_violations(self) -> None:
        """Test multiple simultaneous violations."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        grid_state = GridState(
            voltage_l1=195.0,  # Too low
            current_l1=10.0,
            frequency=61.0,  # Too high
            active_power=2300.0,
            reactive_power=0.0,
        )

        result = grid.grid_code_compliance(grid_state)
        assert result.compliant is False
        assert len(result.violations) >= 2


class TestReactivePowerControl:
    """Test reactive_power_control method."""

    def test_fixed_pf_mode(self) -> None:
        """Test fixed power factor mode."""
        config = ReactivePowerControlConfig(
            mode="fixed_pf", target_power_factor=0.95
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2300.0)

        # Q = P * tan(arccos(pf))
        expected_q = 2300.0 * np.tan(np.arccos(0.95))
        assert abs(q_setpoint - expected_q) < 1.0

    def test_fixed_q_mode(self) -> None:
        """Test fixed reactive power mode."""
        config = ReactivePowerControlConfig(
            mode="fixed_q", target_reactive_power=500.0
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2300.0)
        assert abs(q_setpoint - 500.0) < 0.1

    def test_volt_var_mode_low_voltage(self) -> None:
        """Test volt-var mode with low voltage (absorb Q)."""
        config = ReactivePowerControlConfig(mode="volt_var")
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=207.0,  # 0.90 p.u. - low voltage
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2300.0)
        # Low voltage should generate positive Q (capacitive)
        assert q_setpoint > 0

    def test_volt_var_mode_high_voltage(self) -> None:
        """Test volt-var mode with high voltage (inject Q)."""
        config = ReactivePowerControlConfig(mode="volt_var")
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=253.0,  # 1.10 p.u. - high voltage
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2300.0)
        # High voltage should generate negative Q (inductive)
        assert q_setpoint < 0

    def test_volt_var_mode_nominal_voltage(self) -> None:
        """Test volt-var mode with nominal voltage (no Q)."""
        config = ReactivePowerControlConfig(mode="volt_var")
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,  # 1.0 p.u. - nominal
            current_l1=10.0,
            frequency=60.0,
            active_power=2300.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2300.0)
        # Nominal voltage should generate ~0 Q
        assert abs(q_setpoint) < 100.0

    def test_reactive_power_limiting(self) -> None:
        """Test that reactive power is limited by inverter capability."""
        config = ReactivePowerControlConfig(
            mode="fixed_q", target_reactive_power=10000.0  # Exceeds capability
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            reactive_power_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=20.0,
            frequency=60.0,
            active_power=4000.0,
            reactive_power=0.0,
        )

        q_setpoint = grid.reactive_power_control(grid_state, pv_power=4000.0)

        # Q should be limited by: sqrt(S_max^2 - P^2)
        expected_q_max = np.sqrt(5000.0**2 - 4000.0**2)
        assert q_setpoint <= expected_q_max


class TestFrequencyRegulation:
    """Test frequency_regulation method."""

    def test_nominal_frequency(self) -> None:
        """Test that nominal frequency returns available power."""
        config = FrequencyRegulationConfig(
            enabled=True, nominal_frequency=60.0, droop=0.05, deadband=0.036
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            frequency_regulation_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.0,  # Nominal
            active_power=2300.0,
            reactive_power=0.0,
        )

        p_setpoint = grid.frequency_regulation(grid_state, available_power=2500.0)
        # At nominal frequency, should return available power
        assert abs(p_setpoint - 2500.0) < 100.0

    def test_high_frequency_curtailment(self) -> None:
        """Test that high frequency reduces power output."""
        config = FrequencyRegulationConfig(
            enabled=True, nominal_frequency=60.0, droop=0.05, deadband=0.036
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            frequency_regulation_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.5,  # High frequency
            active_power=2300.0,
            reactive_power=0.0,
        )

        p_setpoint = grid.frequency_regulation(grid_state, available_power=2500.0)
        # High frequency should reduce power
        assert p_setpoint < 2500.0

    def test_low_frequency_increase(self) -> None:
        """Test frequency regulation with low frequency."""
        config = FrequencyRegulationConfig(
            enabled=True, nominal_frequency=60.0, droop=0.05, deadband=0.036
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            frequency_regulation_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=59.5,  # Low frequency
            active_power=2300.0,
            reactive_power=0.0,
        )

        p_setpoint = grid.frequency_regulation(grid_state, available_power=2500.0)
        # Low frequency should try to increase power (but limited by available)
        # Setpoint can't exceed available power
        assert p_setpoint <= 2500.0

    def test_frequency_deadband(self) -> None:
        """Test that frequency deadband works correctly."""
        config = FrequencyRegulationConfig(
            enabled=True, nominal_frequency=60.0, droop=0.05, deadband=0.1
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            frequency_regulation_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.05,  # Within deadband
            active_power=2300.0,
            reactive_power=0.0,
        )

        p_setpoint = grid.frequency_regulation(grid_state, available_power=2500.0)
        # Within deadband, should return available power
        assert abs(p_setpoint - 2500.0) < 100.0

    def test_disabled_frequency_regulation(self) -> None:
        """Test that disabled frequency regulation returns available power."""
        config = FrequencyRegulationConfig(enabled=False)
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            frequency_regulation_config=config,
        )

        grid_state = GridState(
            voltage_l1=230.0,
            current_l1=10.0,
            frequency=60.5,  # High frequency
            active_power=2300.0,
            reactive_power=0.0,
        )

        p_setpoint = grid.frequency_regulation(grid_state, available_power=2500.0)
        # Disabled regulation should return available power unchanged
        assert abs(p_setpoint - 2500.0) < 0.1


class TestSmartGridCommunication:
    """Test smart_grid_communication method."""

    def test_no_scada_config(self) -> None:
        """Test communication without SCADA configuration."""
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
        )

        result = grid.smart_grid_communication({"telemetry": {}})
        assert result["success"] is False
        assert "SCADA configuration not provided" in result["errors"]

    def test_modbus_tcp_communication(self) -> None:
        """Test Modbus TCP communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.100",
            port=502,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {
            "active_power": 2400.0,
            "reactive_power": 100.0,
            "voltage": 235.0,
            "frequency": 50.02,
        }

        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "modbus_tcp"
        assert "sent_data" in result
        assert result["sent_data"] == telemetry

    def test_dnp3_communication(self) -> None:
        """Test DNP3 communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.DNP3,
            host="192.168.1.100",
            port=20000,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {"active_power": 2400.0}
        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "dnp3"

    def test_iec61850_communication(self) -> None:
        """Test IEC 61850 communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.IEC_61850,
            host="192.168.1.100",
            port=102,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {"active_power": 2400.0, "voltage": 235.0}
        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "iec_61850"
        assert "received_data" in result

    def test_opcua_communication(self) -> None:
        """Test OPC UA communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.OPC_UA,
            host="opc.tcp://192.168.1.100",
            port=4840,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {"active_power": 2400.0}
        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "opc_ua"

    def test_mqtt_communication(self) -> None:
        """Test MQTT communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.MQTT,
            host="mqtt.example.com",
            port=1883,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {"active_power": 2400.0}
        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "mqtt"

    def test_sunspec_communication(self) -> None:
        """Test SunSpec Modbus communication."""
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.SUNSPEC,
            host="192.168.1.100",
            port=502,
            device_id="INV001",
        )
        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        telemetry = {"active_power": 2400.0}
        result = grid.smart_grid_communication({"telemetry": telemetry})
        assert result["success"] is True
        assert result["protocol"] == "sunspec"
        assert "received_data" in result


@pytest.mark.integration
class TestGridInteractionIntegration:
    """Integration tests for GridInteraction system."""

    def test_complete_workflow(self) -> None:
        """Test complete grid interaction workflow."""
        # Initialize system
        scada_config = SCADAConfig(
            protocol=SCADAProtocol.MODBUS_TCP,
            host="192.168.1.100",
            port=502,
            device_id="INV001",
        )

        grid = GridInteraction(
            grid_code_standard=GridCodeStandard.IEEE_1547,
            nominal_voltage=230.0,
            rated_power=5000.0,
            scada_config=scada_config,
        )

        # Create grid state
        grid_state = GridState(
            voltage_l1=235.0,
            current_l1=10.5,
            frequency=60.02,
            active_power=2400.0,
            reactive_power=100.0,
            voltage_thd=2.0,
            current_thd=3.0,
            power_factor=0.99,
        )

        # 1. Check compliance
        compliance = grid.grid_code_compliance(grid_state)
        assert compliance.compliant is True

        # 2. Calculate reactive power setpoint
        q_setpoint = grid.reactive_power_control(grid_state, pv_power=2400.0)
        assert isinstance(q_setpoint, float)

        # 3. Calculate active power setpoint (frequency regulation)
        p_setpoint = grid.frequency_regulation(grid_state, available_power=2400.0)
        assert isinstance(p_setpoint, float)
        assert 0 <= p_setpoint <= 2400.0

        # 4. Send data to SCADA
        telemetry = {
            "active_power": p_setpoint,
            "reactive_power": q_setpoint,
            "voltage": grid_state.voltage_l1,
            "frequency": grid_state.frequency,
        }
        scada_result = grid.smart_grid_communication({"telemetry": telemetry})
        assert scada_result["success"] is True
