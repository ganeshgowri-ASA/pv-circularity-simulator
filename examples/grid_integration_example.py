"""
Grid Integration Example.

Demonstrates the use of GridInteraction for PV system grid integration,
including grid code compliance checking, reactive power control,
frequency regulation, and SCADA communication.
"""

from datetime import datetime

from pv_circularity_simulator.grid.grid_interaction import (
    FrequencyRegulationConfig,
    GridCodeStandard,
    GridInteraction,
    GridState,
    ReactivePowerControlConfig,
    SCADAConfig,
    SCADAProtocol,
)


def main() -> None:
    """Run grid integration example."""
    print("=" * 70)
    print("PV Circularity Simulator - Grid Integration Example")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Initialize GridInteraction System
    # =========================================================================
    print("1. Initializing GridInteraction System")
    print("-" * 70)

    # Configure reactive power control (Volt-VAR mode)
    reactive_config = ReactivePowerControlConfig(
        mode="volt_var",
        volt_var_v1=0.92,  # Low voltage threshold
        volt_var_v2=0.98,  # Low voltage deadband
        volt_var_v3=1.02,  # High voltage deadband
        volt_var_v4=1.08,  # High voltage threshold
        volt_var_q1=0.44,  # Max capacitive (absorb Q at low V)
        volt_var_q2=0.0,  # No Q in deadband
        volt_var_q3=0.0,  # No Q in deadband
        volt_var_q4=-0.44,  # Max inductive (inject Q at high V)
    )

    # Configure frequency regulation (droop control)
    frequency_config = FrequencyRegulationConfig(
        enabled=True,
        droop=0.05,  # 5% droop
        deadband=0.036,  # ±0.036 Hz deadband
        nominal_frequency=60.0,  # 60 Hz nominal
        max_power_ramp_rate=0.2,  # 20% per second
    )

    # Configure SCADA communication (Modbus TCP)
    scada_config = SCADAConfig(
        protocol=SCADAProtocol.MODBUS_TCP,
        host="192.168.1.100",
        port=502,
        device_id="PV_INV_001",
        polling_interval=1.0,
        timeout=5.0,
        retry_attempts=3,
        enable_encryption=True,
    )

    # Create GridInteraction instance
    grid = GridInteraction(
        grid_code_standard=GridCodeStandard.IEEE_1547,
        nominal_voltage=230.0,  # 230V single-phase
        rated_power=5000.0,  # 5 kW inverter
        reactive_power_config=reactive_config,
        frequency_regulation_config=frequency_config,
        scada_config=scada_config,
    )

    print(f"Grid Code Standard: {grid.grid_code_standard.value}")
    print(f"Nominal Voltage: {grid.nominal_voltage} V")
    print(f"Rated Power: {grid.rated_power} W")
    print(f"Reactive Power Mode: {grid.reactive_power_config.mode}")
    print(f"Frequency Regulation: {'Enabled' if grid.frequency_regulation_config.enabled else 'Disabled'}")
    print(f"SCADA Protocol: {grid.scada_config.protocol.value if grid.scada_config else 'None'}")
    print()

    # =========================================================================
    # 2. Grid Code Compliance Check - Normal Operation
    # =========================================================================
    print("2. Grid Code Compliance Check - Normal Operation")
    print("-" * 70)

    grid_state_normal = GridState(
        timestamp=datetime.now(),
        voltage_l1=235.0,  # 1.022 p.u. - within limits
        current_l1=10.5,
        frequency=60.02,  # Within limits
        active_power=2400.0,
        reactive_power=100.0,
        voltage_thd=2.0,  # Within 5% limit
        current_thd=3.0,  # Within 5% limit
        power_factor=0.99,  # Above 0.95 limit
    )

    compliance_result = grid.grid_code_compliance(grid_state_normal)

    print(f"Voltage: {grid_state_normal.voltage_l1} V ({grid_state_normal.voltage_l1/grid.nominal_voltage:.3f} p.u.)")
    print(f"Frequency: {grid_state_normal.frequency} Hz")
    print(f"Active Power: {grid_state_normal.active_power} W")
    print(f"Reactive Power: {grid_state_normal.reactive_power} VAR")
    print(f"Voltage THD: {grid_state_normal.voltage_thd}%")
    print(f"Current THD: {grid_state_normal.current_thd}%")
    print(f"Power Factor: {grid_state_normal.power_factor}")
    print()
    print(f"Compliance Status: {'✓ COMPLIANT' if compliance_result.compliant else '✗ NON-COMPLIANT'}")
    if compliance_result.violations:
        print("Violations:")
        for violation in compliance_result.violations:
            print(f"  - {violation}")
    if compliance_result.warnings:
        print("Warnings:")
        for warning in compliance_result.warnings:
            print(f"  - {warning}")
    print()

    # =========================================================================
    # 3. Grid Code Compliance Check - Voltage Violation
    # =========================================================================
    print("3. Grid Code Compliance Check - Voltage Violation")
    print("-" * 70)

    grid_state_high_voltage = GridState(
        voltage_l1=260.0,  # 1.13 p.u. - EXCEEDS 1.10 limit
        current_l1=10.5,
        frequency=60.0,
        active_power=2400.0,
        reactive_power=100.0,
    )

    compliance_result_violation = grid.grid_code_compliance(grid_state_high_voltage)

    print(f"Voltage: {grid_state_high_voltage.voltage_l1} V ({grid_state_high_voltage.voltage_l1/grid.nominal_voltage:.3f} p.u.)")
    print(f"Compliance Status: {'✓ COMPLIANT' if compliance_result_violation.compliant else '✗ NON-COMPLIANT'}")
    if compliance_result_violation.violations:
        print("Violations:")
        for violation in compliance_result_violation.violations:
            print(f"  - {violation}")
    print()

    # =========================================================================
    # 4. Reactive Power Control - Volt-VAR Mode
    # =========================================================================
    print("4. Reactive Power Control - Volt-VAR Mode")
    print("-" * 70)

    # Scenario A: Low voltage - absorb reactive power (capacitive)
    grid_state_low_v = GridState(
        voltage_l1=210.0,  # 0.913 p.u. - low voltage
        current_l1=11.0,
        frequency=60.0,
        active_power=2300.0,
        reactive_power=0.0,
    )

    q_setpoint_low_v = grid.reactive_power_control(grid_state_low_v, pv_power=2300.0)

    print(f"Scenario A: Low Voltage ({grid_state_low_v.voltage_l1/grid.nominal_voltage:.3f} p.u.)")
    print(f"  Reactive Power Setpoint: {q_setpoint_low_v:.2f} VAR (capacitive)")
    print()

    # Scenario B: High voltage - inject reactive power (inductive)
    grid_state_high_v = GridState(
        voltage_l1=250.0,  # 1.087 p.u. - high voltage
        current_l1=9.5,
        frequency=60.0,
        active_power=2300.0,
        reactive_power=0.0,
    )

    q_setpoint_high_v = grid.reactive_power_control(grid_state_high_v, pv_power=2300.0)

    print(f"Scenario B: High Voltage ({grid_state_high_v.voltage_l1/grid.nominal_voltage:.3f} p.u.)")
    print(f"  Reactive Power Setpoint: {q_setpoint_high_v:.2f} VAR (inductive)")
    print()

    # Scenario C: Nominal voltage - minimal reactive power
    q_setpoint_nominal = grid.reactive_power_control(grid_state_normal, pv_power=2400.0)

    print(f"Scenario C: Nominal Voltage ({grid_state_normal.voltage_l1/grid.nominal_voltage:.3f} p.u.)")
    print(f"  Reactive Power Setpoint: {q_setpoint_nominal:.2f} VAR")
    print()

    # =========================================================================
    # 5. Frequency Regulation - Droop Control
    # =========================================================================
    print("5. Frequency Regulation - Droop Control")
    print("-" * 70)

    available_power = 3000.0  # PV is producing 3 kW

    # Scenario A: High frequency - curtail power
    grid_state_high_freq = GridState(
        voltage_l1=230.0,
        current_l1=12.0,
        frequency=60.3,  # High frequency
        active_power=2700.0,
        reactive_power=0.0,
    )

    p_setpoint_high_freq = grid.frequency_regulation(grid_state_high_freq, available_power)

    print(f"Scenario A: High Frequency ({grid_state_high_freq.frequency} Hz)")
    print(f"  Available Power: {available_power:.2f} W")
    print(f"  Active Power Setpoint: {p_setpoint_high_freq:.2f} W (curtailed)")
    print(f"  Curtailment: {available_power - p_setpoint_high_freq:.2f} W ({(1 - p_setpoint_high_freq/available_power)*100:.1f}%)")
    print()

    # Scenario B: Low frequency - increase power (if available)
    grid_state_low_freq = GridState(
        voltage_l1=230.0,
        current_l1=8.0,
        frequency=59.7,  # Low frequency
        active_power=1800.0,
        reactive_power=0.0,
    )

    p_setpoint_low_freq = grid.frequency_regulation(grid_state_low_freq, available_power)

    print(f"Scenario B: Low Frequency ({grid_state_low_freq.frequency} Hz)")
    print(f"  Available Power: {available_power:.2f} W")
    print(f"  Active Power Setpoint: {p_setpoint_low_freq:.2f} W")
    print()

    # Scenario C: Nominal frequency - no adjustment
    grid_state_nominal_freq = GridState(
        voltage_l1=230.0,
        current_l1=10.0,
        frequency=60.0,  # Nominal
        active_power=2300.0,
        reactive_power=0.0,
    )

    p_setpoint_nominal_freq = grid.frequency_regulation(grid_state_nominal_freq, available_power)

    print(f"Scenario C: Nominal Frequency ({grid_state_nominal_freq.frequency} Hz)")
    print(f"  Available Power: {available_power:.2f} W")
    print(f"  Active Power Setpoint: {p_setpoint_nominal_freq:.2f} W (no adjustment)")
    print()

    # =========================================================================
    # 6. Smart Grid Communication - SCADA
    # =========================================================================
    print("6. Smart Grid Communication - SCADA (Modbus TCP)")
    print("-" * 70)

    # Prepare telemetry data
    telemetry_data = {
        "active_power": p_setpoint_nominal_freq,
        "reactive_power": q_setpoint_nominal,
        "voltage": grid_state_normal.voltage_l1,
        "frequency": grid_state_normal.frequency,
        "voltage_thd": grid_state_normal.voltage_thd,
        "current_thd": grid_state_normal.current_thd,
        "power_factor": grid_state_normal.power_factor,
        "connection_state": grid_state_normal.connection_state.value,
    }

    # Communicate with SCADA
    scada_result = grid.smart_grid_communication({"telemetry": telemetry_data})

    print(f"SCADA Communication Status: {'✓ SUCCESS' if scada_result['success'] else '✗ FAILED'}")
    print(f"Protocol: {scada_result['protocol']}")
    print(f"Device ID: {scada_result.get('device_id', 'N/A')}")
    print(f"Timestamp: {scada_result['timestamp']}")
    print()
    print("Sent Telemetry Data:")
    for key, value in scada_result['sent_data'].items():
        print(f"  {key}: {value}")
    print()
    print("Received Control Commands:")
    for key, value in scada_result['received_data'].items():
        print(f"  {key}: {value}")
    print()

    if scada_result['errors']:
        print("Errors:")
        for error in scada_result['errors']:
            print(f"  - {error}")
        print()

    # =========================================================================
    # 7. Complete Workflow Summary
    # =========================================================================
    print("7. Complete Workflow Summary")
    print("-" * 70)
    print(f"Grid Code Standard: {grid.grid_code_standard.value}")
    print(f"Grid Compliance: {'✓ PASS' if compliance_result.compliant else '✗ FAIL'}")
    print(f"Active Power Setpoint: {p_setpoint_nominal_freq:.2f} W")
    print(f"Reactive Power Setpoint: {q_setpoint_nominal:.2f} VAR")
    print(f"Frequency Regulation: {'Enabled' if grid.frequency_regulation_config.enabled else 'Disabled'}")
    print(f"SCADA Communication: {'✓ Connected' if scada_result['success'] else '✗ Disconnected'}")
    print()

    print("=" * 70)
    print("Grid Integration Example Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
