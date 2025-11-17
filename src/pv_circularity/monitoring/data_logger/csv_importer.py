"""
CSV file importer for historical data and batch imports.

This module provides functionality to import monitoring data from CSV files,
supporting various formats and automatic column mapping.
"""

import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from dateutil import parser as date_parser

from pv_circularity.core import get_logger, DataValidationError
from pv_circularity.core.utils import get_utc_now, to_utc
from pv_circularity.models.monitoring import MonitoringDataPoint

logger = get_logger(__name__)


class CSVFileImporter:
    """
    Import monitoring data from CSV files.

    Supports various CSV formats with automatic column detection and mapping.
    Can handle timestamp parsing, unit conversion, and data validation.

    Args:
        file_path: Path to CSV file
        device_id: Device identifier for imported data
        timestamp_column: Name of timestamp column
        value_columns: Mapping of parameter names to column names
        unit_mapping: Mapping of parameters to units

    Example:
        >>> importer = CSVFileImporter(
        ...     file_path="data/inverter_data.csv",
        ...     device_id="INV001",
        ...     timestamp_column="timestamp",
        ...     value_columns={"ac_power": "AC Power (kW)", "dc_voltage": "DC Voltage (V)"},
        ...     unit_mapping={"ac_power": "kW", "dc_voltage": "V"}
        ... )
        >>> data_points = await importer.import_data()
        >>> print(f"Imported {len(data_points)} data points")
    """

    def __init__(
        self,
        file_path: Path | str,
        device_id: str,
        timestamp_column: str = "timestamp",
        value_columns: Optional[Dict[str, str]] = None,
        unit_mapping: Optional[Dict[str, str]] = None,
        date_format: Optional[str] = None,
    ) -> None:
        """
        Initialize CSV file importer.

        Args:
            file_path: Path to CSV file
            device_id: Device identifier
            timestamp_column: Name of timestamp column
            value_columns: Mapping {parameter: column_name}
            unit_mapping: Mapping {parameter: unit}
            date_format: Date format string (e.g., "%Y-%m-%d %H:%M:%S")
        """
        self.file_path = Path(file_path)
        self.device_id = device_id
        self.timestamp_column = timestamp_column
        self.value_columns = value_columns or {}
        self.unit_mapping = unit_mapping or {}
        self.date_format = date_format

        logger.info(
            "CSVFileImporter initialized",
            file_path=str(self.file_path),
            device_id=device_id,
        )

    async def import_data(
        self, skip_errors: bool = True, max_rows: Optional[int] = None
    ) -> List[MonitoringDataPoint]:
        """
        Import data from CSV file.

        Args:
            skip_errors: Continue importing on errors (skip bad rows)
            max_rows: Maximum number of rows to import (None for all)

        Returns:
            List of imported monitoring data points

        Raises:
            FileNotFoundError: If CSV file not found
            DataValidationError: If data format is invalid and skip_errors is False
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        logger.info("Starting CSV import", file_path=str(self.file_path))

        try:
            # Read CSV file using pandas
            df = pd.read_csv(self.file_path, nrows=max_rows)

            logger.debug(
                "CSV file loaded",
                rows=len(df),
                columns=len(df.columns),
            )

            # Validate required columns
            if self.timestamp_column not in df.columns:
                raise DataValidationError(
                    f"Timestamp column '{self.timestamp_column}' not found in CSV",
                    field="timestamp_column",
                )

            # Auto-detect value columns if not provided
            if not self.value_columns:
                self.value_columns = self._auto_detect_value_columns(df)
                logger.info(
                    "Auto-detected value columns",
                    columns=list(self.value_columns.keys()),
                )

            # Convert to monitoring data points
            data_points = []
            errors = 0

            for idx, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = self._parse_timestamp(row[self.timestamp_column])

                    # Extract values for each parameter
                    for parameter, column_name in self.value_columns.items():
                        if column_name not in df.columns:
                            if not skip_errors:
                                raise DataValidationError(
                                    f"Column '{column_name}' not found",
                                    field="column_name",
                                )
                            continue

                        value = row[column_name]

                        # Skip if value is NaN or None
                        if pd.isna(value):
                            continue

                        # Create data point
                        data_point = MonitoringDataPoint(
                            device_id=self.device_id,
                            timestamp=timestamp,
                            parameter=parameter,
                            value=float(value),
                            unit=self.unit_mapping.get(parameter, ""),
                            quality=1.0,
                            metadata={"source": "csv_import", "row_index": int(idx)},
                        )
                        data_points.append(data_point)

                except Exception as e:
                    errors += 1
                    if not skip_errors:
                        raise DataValidationError(
                            f"Error processing row {idx}: {str(e)}",
                            details={"row_index": idx, "error": str(e)},
                        )
                    logger.warning(
                        "Skipping row due to error",
                        row_index=idx,
                        error=str(e),
                    )

            logger.info(
                "CSV import complete",
                total_points=len(data_points),
                errors=errors,
            )

            return data_points

        except Exception as e:
            logger.error("CSV import failed", error=str(e), exc_info=True)
            raise

    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """
        Parse timestamp from various formats.

        Args:
            timestamp_str: Timestamp string or value

        Returns:
            Parsed datetime object (timezone-aware)

        Raises:
            DataValidationError: If timestamp cannot be parsed
        """
        try:
            if isinstance(timestamp_str, datetime):
                return to_utc(timestamp_str)

            if self.date_format:
                dt = datetime.strptime(str(timestamp_str), self.date_format)
            else:
                # Use dateutil parser for flexible parsing
                dt = date_parser.parse(str(timestamp_str))

            return to_utc(dt)

        except Exception as e:
            raise DataValidationError(
                f"Failed to parse timestamp: {timestamp_str}",
                field="timestamp",
                value=timestamp_str,
                original_exception=e,
            )

    def _auto_detect_value_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect numeric value columns from DataFrame.

        Args:
            df: pandas DataFrame

        Returns:
            Dictionary mapping parameter names to column names
        """
        value_columns = {}

        # Skip timestamp column and detect numeric columns
        for column in df.columns:
            if column == self.timestamp_column:
                continue

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                # Use column name as parameter name (cleaned up)
                parameter = column.lower().replace(" ", "_").replace("(", "").replace(")", "")
                value_columns[parameter] = column

        return value_columns

    def get_column_info(self) -> Dict[str, Any]:
        """
        Get information about columns in the CSV file.

        Returns:
            Dictionary with column information

        Raises:
            FileNotFoundError: If CSV file not found
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        df = pd.read_csv(self.file_path, nrows=5)

        return {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
