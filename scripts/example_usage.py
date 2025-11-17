"""
Example usage of PV Circularity Simulator monitoring system.

This script demonstrates how to use the real-time monitoring, SCADA integration,
and data logging capabilities.
"""

import asyncio
from pathlib import Path

from pv_circularity.core import get_logger, setup_logging
from pv_circularity.models.scada import (
    SCADADevice,
    ModbusConfig,
    ProtocolType,
    DeviceType,
)
from pv_circularity.monitoring import (
    DataLoggerIntegrator,
    SCADAConnector,
    DataAggregator,
    RealTimeMonitor,
    PerformanceMetrics,
    AlertEngine,
)

# Setup logging
setup_logging(log_level="INFO", log_format="console")
logger = get_logger(__name__)


async def main():
    """Main example function demonstrating the monitoring system."""

    logger.info("Starting PV Circularity Simulator example")

    # Example 1: Configure SCADA devices
    logger.info("=== Example 1: Configure SCADA Devices ===")

    # Create a Modbus TCP device
    modbus_device = SCADADevice(
        device_id="INV001",
        name="Main Inverter",
        device_type=DeviceType.INVERTER,
        protocol_type=ProtocolType.MODBUS_TCP,
        modbus_config=ModbusConfig(
            host="192.168.1.100",
            port=502,
            slave_id=1,
            register_map={
                "ac_power": {
                    "address": 30775,
                    "count": 2,
                    "type": "input",
                    "scale": 1.0,
                    "unit": "W",
                },
                "dc_voltage": {
                    "address": 30771,
                    "count": 1,
                    "type": "input",
                    "scale": 0.1,
                    "unit": "V",
                },
            },
        ),
        enabled=True,
        poll_interval_seconds=5,
        site_id="SITE001",
    )

    devices = [modbus_device]

    # Example 2: Initialize Data Logger Integrator
    logger.info("=== Example 2: Data Logger Integrator ===")

    integrator = DataLoggerIntegrator(devices)
    # Note: In real usage, you would call await integrator.initialize()
    # to connect to actual devices

    logger.info(f"Integrator initialized with {len(devices)} devices")

    # Example 3: SCADA Connector
    logger.info("=== Example 3: SCADA Connector ===")

    connector = SCADAConnector(devices)
    # In real usage:
    # await connector.connect()
    # data = await connector.read_all_devices()

    logger.info("SCADA connector created")

    # Example 4: Data Aggregator
    logger.info("=== Example 4: Data Aggregator ===")

    aggregator = DataAggregator()

    # Example data aggregation
    logger.info("Data aggregator ready for multi-site aggregation")

    # Example 5: Real-Time Monitor
    logger.info("=== Example 5: Real-Time Monitor ===")

    monitor = RealTimeMonitor(devices, update_interval=5)

    # In real usage:
    # await monitor.start_monitoring()
    # async for data_batch in monitor.live_data_stream():
    #     logger.info(f"Received {len(data_batch)} data points")

    logger.info("Real-time monitor configured")

    # Example 6: Performance Metrics
    logger.info("=== Example 6: Performance Metrics ===")

    metrics = PerformanceMetrics()

    # Calculate instantaneous PR
    pr = await metrics.instantaneous_pr(
        actual_power=85.5,
        rated_power=100.0,
        irradiance=850,
        reference_irradiance=1000,
    )
    logger.info(f"Instantaneous Performance Ratio: {pr:.2%}")

    # Calculate capacity factor
    cf = await metrics.capacity_factor(
        actual_energy=5000,
        rated_power=1000,
        period_hours=24,
    )
    logger.info(f"Capacity Factor: {cf:.2%}")

    # Calculate specific yield
    sy = await metrics.specific_yield(
        actual_energy=5000,
        rated_power=1000,
    )
    logger.info(f"Specific Yield: {sy:.2f} kWh/kWp")

    # Example 7: Alert Engine
    logger.info("=== Example 7: Alert Engine ===")

    alert_engine = AlertEngine(underperformance_threshold=15.0)

    # Subscribe to alerts
    def on_alert(alert):
        logger.warning(f"ALERT: {alert.message} (Severity: {alert.severity})")

    alert_engine.subscribe_alerts(on_alert)

    # In real usage:
    # await alert_engine.start()
    # ... monitoring will trigger alerts automatically
    # await alert_engine.stop()

    logger.info("Alert engine configured")

    # Example 8: Underperformance Detection
    logger.info("=== Example 8: Underperformance Detection ===")

    alert = await alert_engine.underperformance_detection(
        site_id="SITE001",
        device_id="INV001",
        actual_power=80,
        expected_power=100,
    )

    if alert:
        logger.warning(f"Underperformance alert: {alert.message}")
    else:
        logger.info("No underperformance detected")

    # Example 9: CSV Import
    logger.info("=== Example 9: CSV File Import ===")

    from pv_circularity.monitoring.data_logger import CSVFileImporter

    # Note: This would require an actual CSV file to exist
    logger.info("CSV importer can import historical data from files")

    # Example usage (commented as file may not exist):
    # csv_importer = CSVFileImporter(
    #     file_path="data/inverter_data.csv",
    #     device_id="INV001",
    #     timestamp_column="timestamp",
    #     value_columns={
    #         "ac_power": "AC Power (kW)",
    #         "dc_voltage": "DC Voltage (V)",
    #     },
    # )
    # data_points = await csv_importer.import_data()

    logger.info("PV Circularity Simulator example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
