"""
InfluxDB client for time-series data storage.

This module provides a client for storing and retrieving time-series monitoring
data using InfluxDB.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import asyncio

from config.settings import Settings
from src.core.models.schemas import (
    InverterData,
    StringData,
    ModuleData,
    SCADAData,
    PerformanceRatioData
)

logger = logging.getLogger(__name__)


class InfluxDBTimeSeriesClient:
    """
    Client for InfluxDB time-series database operations.

    Handles writing and querying time-series monitoring data including inverter
    telemetry, SCADA data, and performance metrics.

    Attributes:
        settings: Application settings
        _client: InfluxDB client instance
        _write_api: Write API instance
        _query_api: Query API instance
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize InfluxDB client.

        Args:
            settings: Application settings containing InfluxDB configuration.
        """
        self.settings = settings
        self._client: Optional[InfluxDBClient] = None
        self._write_api = None
        self._query_api = None

    async def connect(self) -> None:
        """
        Connect to InfluxDB server.

        Raises:
            ConnectionError: If unable to connect to InfluxDB.
        """
        try:
            self._client = InfluxDBClient(
                url=self.settings.influxdb.url,
                token=self.settings.influxdb.token,
                org=self.settings.influxdb.org
            )

            # Create write API with batching
            self._write_api = self._client.write_api(
                write_options=ASYNCHRONOUS if asyncio.iscoroutinefunction(self.write_inverter_data)
                else SYNCHRONOUS
            )

            self._query_api = self._client.query_api()

            # Test connection
            health = self._client.health()
            if health.status != "pass":
                raise ConnectionError(f"InfluxDB health check failed: {health.status}")

            logger.info(f"Connected to InfluxDB at {self.settings.influxdb.url}")

        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {e}", exc_info=True)
            raise ConnectionError(f"InfluxDB connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from InfluxDB server."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from InfluxDB")

    async def write_inverter_data(self, data: InverterData) -> None:
        """
        Write inverter data to InfluxDB.

        Args:
            data: InverterData instance to write.
        """
        point = (
            Point("inverter")
            .tag("inverter_id", data.inverter_id)
            .tag("site_id", self.settings.site_id)
            .tag("status", data.status)
            .field("dc_power", float(data.dc_power))
            .field("ac_power", float(data.ac_power))
            .field("dc_voltage", float(data.dc_voltage))
            .field("dc_current", float(data.dc_current))
            .field("ac_voltage_l1", float(data.ac_voltage_l1))
            .field("ac_current_l1", float(data.ac_current_l1))
            .field("temperature", float(data.temperature))
            .field("energy_daily", float(data.energy_daily))
            .field("energy_total", float(data.energy_total))
            .time(data.timestamp, WritePrecision.NS)
        )

        if data.efficiency is not None:
            point = point.field("efficiency", float(data.efficiency))

        if data.reactive_power is not None:
            point = point.field("reactive_power", float(data.reactive_power))

        if data.power_factor is not None:
            point = point.field("power_factor", float(data.power_factor))

        self._write_api.write(
            bucket=self.settings.influxdb.bucket,
            org=self.settings.influxdb.org,
            record=point
        )

        logger.debug(f"Wrote inverter data for {data.inverter_id}")

    async def write_string_data(self, data: StringData) -> None:
        """
        Write string-level data to InfluxDB.

        Args:
            data: StringData instance to write.
        """
        point = (
            Point("string")
            .tag("string_id", data.string_id)
            .tag("inverter_id", data.inverter_id)
            .tag("site_id", self.settings.site_id)
            .field("voltage", float(data.voltage))
            .field("current", float(data.current))
            .field("power", float(data.power))
            .time(data.timestamp, WritePrecision.NS)
        )

        if data.temperature is not None:
            point = point.field("temperature", float(data.temperature))

        if data.irradiance is not None:
            point = point.field("irradiance", float(data.irradiance))

        self._write_api.write(
            bucket=self.settings.influxdb.bucket,
            org=self.settings.influxdb.org,
            record=point
        )

        logger.debug(f"Wrote string data for {data.string_id}")

    async def write_module_data(self, data: ModuleData) -> None:
        """
        Write module-level data to InfluxDB.

        Args:
            data: ModuleData instance to write.
        """
        point = (
            Point("module")
            .tag("module_id", data.module_id)
            .tag("string_id", data.string_id)
            .tag("site_id", self.settings.site_id)
            .field("voltage", float(data.voltage))
            .field("current", float(data.current))
            .field("power", float(data.power))
            .field("temperature", float(data.temperature))
            .field("hotspot_detected", data.hotspot_detected)
            .time(data.timestamp, WritePrecision.NS)
        )

        if data.efficiency is not None:
            point = point.field("efficiency", float(data.efficiency))

        if data.max_cell_temperature is not None:
            point = point.field("max_cell_temperature", float(data.max_cell_temperature))

        self._write_api.write(
            bucket=self.settings.influxdb.bucket,
            org=self.settings.influxdb.org,
            record=point
        )

        logger.debug(f"Wrote module data for {data.module_id}")

    async def write_scada_data(self, data: SCADAData) -> None:
        """
        Write SCADA data to InfluxDB.

        Args:
            data: SCADAData instance to write.
        """
        point = (
            Point("scada")
            .tag("site_id", data.site_id)
            .field("total_dc_power", float(data.total_dc_power))
            .field("total_ac_power", float(data.total_ac_power))
            .field("irradiance", float(data.irradiance))
            .field("ambient_temperature", float(data.ambient_temperature))
            .field("available_inverters", data.available_inverters)
            .field("total_inverters", data.total_inverters)
            .time(data.timestamp, WritePrecision.NS)
        )

        if data.module_temperature is not None:
            point = point.field("module_temperature", float(data.module_temperature))

        if data.wind_speed is not None:
            point = point.field("wind_speed", float(data.wind_speed))

        if data.grid_frequency is not None:
            point = point.field("grid_frequency", float(data.grid_frequency))

        if data.grid_voltage is not None:
            point = point.field("grid_voltage", float(data.grid_voltage))

        self._write_api.write(
            bucket=self.settings.influxdb.bucket,
            org=self.settings.influxdb.org,
            record=point
        )

        logger.debug(f"Wrote SCADA data for {data.site_id}")

    async def write_performance_ratio(self, data: PerformanceRatioData) -> None:
        """
        Write Performance Ratio data to InfluxDB.

        Args:
            data: PerformanceRatioData instance to write.
        """
        point = (
            Point("performance_ratio")
            .tag("site_id", data.site_id)
            .field("instantaneous_pr", float(data.instantaneous_pr))
            .field("actual_energy", float(data.actual_energy))
            .field("expected_energy", float(data.expected_energy))
            .field("actual_irradiance", float(data.actual_irradiance))
            .field("actual_temperature", float(data.actual_temperature))
            .time(data.timestamp, WritePrecision.NS)
        )

        self._write_api.write(
            bucket=self.settings.influxdb.bucket,
            org=self.settings.influxdb.org,
            record=point
        )

        logger.debug(f"Wrote PR data: {data.instantaneous_pr:.2f}%")

    async def query_inverter_data(
        self,
        inverter_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query inverter data from InfluxDB.

        Args:
            inverter_id: Inverter ID to query
            start_time: Start time for query
            end_time: End time for query (defaults to now)

        Returns:
            List of data points as dictionaries.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        query = f'''
        from(bucket: "{self.settings.influxdb.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r._measurement == "inverter")
            |> filter(fn: (r) => r.inverter_id == "{inverter_id}")
        '''

        tables = self._query_api.query(query, org=self.settings.influxdb.org)

        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    'time': record.get_time(),
                    'field': record.get_field(),
                    'value': record.get_value(),
                    'inverter_id': record.values.get('inverter_id')
                })

        return results

    async def query_aggregated_power(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        window: str = "1m"
    ) -> List[Dict[str, Any]]:
        """
        Query aggregated power data.

        Args:
            start_time: Start time for query
            end_time: End time for query (defaults to now)
            window: Aggregation window (e.g., "1m", "5m", "1h")

        Returns:
            List of aggregated data points.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        query = f'''
        from(bucket: "{self.settings.influxdb.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r._measurement == "inverter")
            |> filter(fn: (r) => r._field == "ac_power")
            |> aggregateWindow(every: {window}, fn: mean)
            |> sum()
        '''

        tables = self._query_api.query(query, org=self.settings.influxdb.org)

        results = []
        for table in tables:
            for record in table.records:
                results.append({
                    'time': record.get_time(),
                    'total_power': record.get_value()
                })

        return results

    async def query_performance_metrics(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Query performance metrics summary.

        Args:
            start_time: Start time for query
            end_time: End time for query (defaults to now)

        Returns:
            Dictionary containing performance summary metrics.
        """
        if end_time is None:
            end_time = datetime.utcnow()

        # Query average PR
        pr_query = f'''
        from(bucket: "{self.settings.influxdb.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r._measurement == "performance_ratio")
            |> filter(fn: (r) => r._field == "instantaneous_pr")
            |> mean()
        '''

        pr_tables = self._query_api.query(pr_query, org=self.settings.influxdb.org)
        avg_pr = None
        for table in pr_tables:
            for record in table.records:
                avg_pr = record.get_value()
                break

        # Query total energy
        energy_query = f'''
        from(bucket: "{self.settings.influxdb.bucket}")
            |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
            |> filter(fn: (r) => r._measurement == "inverter")
            |> filter(fn: (r) => r._field == "energy_daily")
            |> sum()
        '''

        energy_tables = self._query_api.query(energy_query, org=self.settings.influxdb.org)
        total_energy = None
        for table in energy_tables:
            for record in table.records:
                total_energy = record.get_value()
                break

        return {
            'average_pr': avg_pr,
            'total_energy': total_energy,
            'period_start': start_time,
            'period_end': end_time
        }
