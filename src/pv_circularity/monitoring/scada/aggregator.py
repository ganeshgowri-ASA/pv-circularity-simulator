"""
Data Aggregator for multi-site data aggregation and normalization.

This module provides functionality to aggregate data from multiple sites,
normalize different data formats, and align timestamps.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

from pv_circularity.core import get_logger, AggregationError
from pv_circularity.core.utils import get_utc_now, to_utc
from pv_circularity.models.monitoring import MonitoringDataPoint

logger = get_logger(__name__)


class DataAggregator:
    """
    Aggregate and normalize monitoring data from multiple sites.

    This class provides methods for multi-site aggregation, data normalization,
    timestamp alignment, and statistical aggregation functions.

    Example:
        >>> aggregator = DataAggregator()
        >>> aggregated_data = await aggregator.multi_site_aggregation(site_data_dict)
        >>> normalized_data = await aggregator.data_normalization(raw_data)
        >>> aligned_data = await aggregator.timestamp_alignment(data_points, interval=60)
    """

    def __init__(self) -> None:
        """Initialize data aggregator."""
        logger.info("DataAggregator initialized")

    async def multi_site_aggregation(
        self,
        site_data: Dict[str, List[MonitoringDataPoint]],
        aggregation_method: str = "sum",
    ) -> Dict[str, List[MonitoringDataPoint]]:
        """
        Aggregate data from multiple sites.

        Args:
            site_data: Dictionary mapping site IDs to their data points
            aggregation_method: Method to use ('sum', 'mean', 'max', 'min')

        Returns:
            Dictionary with aggregated data points per parameter

        Example:
            >>> site_data = {
            ...     "SITE001": [data_point1, data_point2],
            ...     "SITE002": [data_point3, data_point4],
            ... }
            >>> aggregated = await aggregator.multi_site_aggregation(site_data, "sum")
        """
        logger.info(
            "Starting multi-site aggregation",
            sites=len(site_data),
            method=aggregation_method,
        )

        try:
            # Group data points by parameter and timestamp
            grouped_data: Dict[str, Dict[datetime, List[float]]] = defaultdict(
                lambda: defaultdict(list)
            )

            for site_id, data_points in site_data.items():
                for dp in data_points:
                    grouped_data[dp.parameter][dp.timestamp].append(dp.value)

            # Aggregate values
            aggregated_points: Dict[str, List[MonitoringDataPoint]] = defaultdict(list)

            for parameter, timestamp_values in grouped_data.items():
                for timestamp, values in timestamp_values.items():
                    # Apply aggregation method
                    if aggregation_method == "sum":
                        agg_value = sum(values)
                    elif aggregation_method == "mean":
                        agg_value = np.mean(values)
                    elif aggregation_method == "max":
                        agg_value = max(values)
                    elif aggregation_method == "min":
                        agg_value = min(values)
                    else:
                        raise AggregationError(
                            f"Unknown aggregation method: {aggregation_method}",
                            details={"method": aggregation_method},
                        )

                    # Get first data point as template
                    first_site_id = list(site_data.keys())[0]
                    template_dp = next(
                        (
                            dp
                            for dp in site_data[first_site_id]
                            if dp.parameter == parameter and dp.timestamp == timestamp
                        ),
                        None,
                    )

                    if template_dp:
                        aggregated_point = MonitoringDataPoint(
                            device_id="AGGREGATED",
                            timestamp=timestamp,
                            parameter=parameter,
                            value=agg_value,
                            unit=template_dp.unit,
                            quality=1.0,
                            metadata={
                                "aggregation_method": aggregation_method,
                                "site_count": len(values),
                            },
                        )
                        aggregated_points[parameter].append(aggregated_point)

            logger.info(
                "Multi-site aggregation complete",
                parameters=len(aggregated_points),
            )

            return dict(aggregated_points)

        except Exception as e:
            logger.error("Multi-site aggregation failed", error=str(e), exc_info=True)
            raise AggregationError(
                f"Multi-site aggregation failed: {str(e)}",
                original_exception=e,
            )

    async def data_normalization(
        self,
        data_points: List[MonitoringDataPoint],
        normalize_units: bool = True,
        fill_missing: bool = True,
    ) -> List[MonitoringDataPoint]:
        """
        Normalize data points to standard format and units.

        Args:
            data_points: List of data points to normalize
            normalize_units: Convert to standard units
            fill_missing: Fill missing values with interpolation

        Returns:
            List of normalized data points

        Example:
            >>> normalized = await aggregator.data_normalization(raw_data_points)
        """
        logger.info(
            "Starting data normalization",
            data_points=len(data_points),
        )

        try:
            normalized_points = []

            # Unit conversion mapping
            unit_conversions = {
                "W": {"target": "kW", "factor": 0.001},
                "Wh": {"target": "kWh", "factor": 0.001},
                "mV": {"target": "V", "factor": 0.001},
                "mA": {"target": "A", "factor": 0.001},
            }

            for dp in data_points:
                normalized_value = dp.value
                normalized_unit = dp.unit

                # Normalize units
                if normalize_units and dp.unit in unit_conversions:
                    conversion = unit_conversions[dp.unit]
                    normalized_value = dp.value * conversion["factor"]
                    normalized_unit = conversion["target"]

                # Create normalized data point
                normalized_point = MonitoringDataPoint(
                    device_id=dp.device_id,
                    timestamp=dp.timestamp,
                    parameter=dp.parameter,
                    value=normalized_value,
                    unit=normalized_unit,
                    quality=dp.quality,
                    metadata={**dp.metadata, "normalized": True},
                )
                normalized_points.append(normalized_point)

            logger.info(
                "Data normalization complete",
                normalized_points=len(normalized_points),
            )

            return normalized_points

        except Exception as e:
            logger.error("Data normalization failed", error=str(e), exc_info=True)
            raise AggregationError(
                f"Data normalization failed: {str(e)}",
                original_exception=e,
            )

    async def timestamp_alignment(
        self,
        data_points: List[MonitoringDataPoint],
        interval_seconds: int = 60,
        method: str = "mean",
    ) -> List[MonitoringDataPoint]:
        """
        Align timestamps to regular intervals using aggregation.

        Args:
            data_points: List of data points to align
            interval_seconds: Alignment interval in seconds
            method: Aggregation method ('mean', 'sum', 'max', 'min', 'last')

        Returns:
            List of timestamp-aligned data points

        Example:
            >>> aligned = await aggregator.timestamp_alignment(data_points, interval=300)
        """
        logger.info(
            "Starting timestamp alignment",
            data_points=len(data_points),
            interval_seconds=interval_seconds,
        )

        try:
            if not data_points:
                return []

            # Convert to pandas DataFrame for easier manipulation
            df_data = []
            for dp in data_points:
                df_data.append(
                    {
                        "device_id": dp.device_id,
                        "timestamp": dp.timestamp,
                        "parameter": dp.parameter,
                        "value": dp.value,
                        "unit": dp.unit,
                        "quality": dp.quality,
                    }
                )

            df = pd.DataFrame(df_data)

            # Group by device and parameter, then resample by time interval
            aligned_points = []

            for (device_id, parameter), group in df.groupby(["device_id", "parameter"]):
                # Set timestamp as index and sort
                group_indexed = group.set_index("timestamp").sort_index()

                # Resample to specified interval
                if method == "mean":
                    resampled = group_indexed["value"].resample(f"{interval_seconds}S").mean()
                elif method == "sum":
                    resampled = group_indexed["value"].resample(f"{interval_seconds}S").sum()
                elif method == "max":
                    resampled = group_indexed["value"].resample(f"{interval_seconds}S").max()
                elif method == "min":
                    resampled = group_indexed["value"].resample(f"{interval_seconds}S").min()
                elif method == "last":
                    resampled = group_indexed["value"].resample(f"{interval_seconds}S").last()
                else:
                    raise AggregationError(
                        f"Unknown alignment method: {method}",
                        details={"method": method},
                    )

                # Get unit from first row
                unit = group_indexed["unit"].iloc[0] if len(group_indexed) > 0 else ""

                # Create aligned data points
                for timestamp, value in resampled.items():
                    if not pd.isna(value):
                        aligned_point = MonitoringDataPoint(
                            device_id=device_id,
                            timestamp=timestamp.to_pydatetime().replace(tzinfo=None),
                            parameter=parameter,
                            value=float(value),
                            unit=unit,
                            quality=1.0,
                            metadata={
                                "aligned": True,
                                "interval_seconds": interval_seconds,
                                "method": method,
                            },
                        )
                        # Ensure timezone-aware
                        aligned_point.timestamp = to_utc(aligned_point.timestamp)
                        aligned_points.append(aligned_point)

            logger.info(
                "Timestamp alignment complete",
                aligned_points=len(aligned_points),
            )

            return aligned_points

        except Exception as e:
            logger.error("Timestamp alignment failed", error=str(e), exc_info=True)
            raise AggregationError(
                f"Timestamp alignment failed: {str(e)}",
                original_exception=e,
            )

    async def aggregate_by_time_window(
        self,
        data_points: List[MonitoringDataPoint],
        window_size: timedelta,
        aggregation_method: str = "mean",
    ) -> List[MonitoringDataPoint]:
        """
        Aggregate data points over time windows.

        Args:
            data_points: List of data points
            window_size: Size of aggregation window
            aggregation_method: Aggregation method

        Returns:
            List of aggregated data points
        """
        # Convert window size to seconds
        window_seconds = int(window_size.total_seconds())
        return await self.timestamp_alignment(
            data_points, interval_seconds=window_seconds, method=aggregation_method
        )
