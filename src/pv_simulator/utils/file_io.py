"""
File I/O Handlers for PV Circularity Simulator.

This module provides comprehensive file input/output utilities for reading
and writing data in various formats (CSV, JSON, YAML) commonly used in
photovoltaic system analysis.
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import yaml

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# Type aliases
PathLike = Union[str, Path]
DataDict = Dict[str, Any]
DataList = List[DataDict]


def read_json(
    file_path: PathLike,
    encoding: str = "utf-8",
    default: Optional[Any] = None,
) -> Any:
    """
    Read data from a JSON file.

    Args:
        file_path: Path to the JSON file
        encoding: File encoding (default: utf-8)
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data (dict, list, or other JSON-serializable type)

    Raises:
        FileNotFoundError: If file doesn't exist and no default provided
        json.JSONDecodeError: If file contains invalid JSON and no default provided

    Examples:
        >>> data = read_json("config.json")
        >>> data = read_json("missing.json", default={})
    """
    path = Path(file_path)

    try:
        with path.open("r", encoding=encoding) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        if default is not None:
            return default
        raise


def write_json(
    data: Any,
    file_path: PathLike,
    encoding: str = "utf-8",
    indent: int = 2,
    ensure_ascii: bool = False,
    create_dirs: bool = True,
) -> None:
    """
    Write data to a JSON file.

    Args:
        data: Data to write (must be JSON-serializable)
        file_path: Path to the output file
        encoding: File encoding (default: utf-8)
        indent: Number of spaces for indentation (default: 2)
        ensure_ascii: If True, escape non-ASCII characters (default: False)
        create_dirs: If True, create parent directories if they don't exist

    Raises:
        TypeError: If data is not JSON-serializable

    Examples:
        >>> write_json({"key": "value"}, "output.json")
        >>> write_json([1, 2, 3], "data.json", indent=4)
    """
    path = Path(file_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_yaml(
    file_path: PathLike,
    encoding: str = "utf-8",
    default: Optional[Any] = None,
) -> Any:
    """
    Read data from a YAML file.

    Args:
        file_path: Path to the YAML file
        encoding: File encoding (default: utf-8)
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        Parsed YAML data (dict, list, or other YAML type)

    Raises:
        FileNotFoundError: If file doesn't exist and no default provided
        yaml.YAMLError: If file contains invalid YAML and no default provided

    Examples:
        >>> config = read_yaml("config.yaml")
        >>> config = read_yaml("missing.yaml", default={})
    """
    path = Path(file_path)

    try:
        with path.open("r", encoding=encoding) as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        if default is not None:
            return default
        raise


def write_yaml(
    data: Any,
    file_path: PathLike,
    encoding: str = "utf-8",
    default_flow_style: bool = False,
    sort_keys: bool = False,
    create_dirs: bool = True,
) -> None:
    """
    Write data to a YAML file.

    Args:
        data: Data to write (must be YAML-serializable)
        file_path: Path to the output file
        encoding: File encoding (default: utf-8)
        default_flow_style: If True, use flow style (inline) formatting
        sort_keys: If True, sort dictionary keys
        create_dirs: If True, create parent directories if they don't exist

    Examples:
        >>> write_yaml({"key": "value"}, "output.yaml")
        >>> write_yaml([1, 2, 3], "data.yaml")
    """
    path = Path(file_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding) as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,
        )


def read_csv(
    file_path: PathLike,
    encoding: str = "utf-8",
    delimiter: str = ",",
    has_header: bool = True,
    skip_rows: int = 0,
) -> List[Dict[str, str]]:
    """
    Read data from a CSV file.

    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: utf-8)
        delimiter: Field delimiter (default: ',')
        has_header: If True, first row contains column names
        skip_rows: Number of rows to skip at the beginning

    Returns:
        List of dictionaries (if has_header=True) or list of lists

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> data = read_csv("data.csv")
        >>> data = read_csv("data.tsv", delimiter="\\t")
    """
    path = Path(file_path)
    rows = []

    with path.open("r", encoding=encoding, newline="") as f:
        # Skip specified number of rows
        for _ in range(skip_rows):
            next(f, None)

        if has_header:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        else:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

    return rows


def write_csv(
    data: Union[DataList, List[List[Any]]],
    file_path: PathLike,
    encoding: str = "utf-8",
    delimiter: str = ",",
    fieldnames: Optional[List[str]] = None,
    write_header: bool = True,
    create_dirs: bool = True,
) -> None:
    """
    Write data to a CSV file.

    Args:
        data: List of dictionaries or list of lists to write
        file_path: Path to the output file
        encoding: File encoding (default: utf-8)
        delimiter: Field delimiter (default: ',')
        fieldnames: Column names (required if data is list of dicts and write_header=True)
        write_header: If True, write header row
        create_dirs: If True, create parent directories if they don't exist

    Raises:
        ValueError: If data format is incompatible with parameters

    Examples:
        >>> data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
        >>> write_csv(data, "output.csv")
        >>> write_csv([[1, 2], [3, 4]], "data.csv", fieldnames=["A", "B"])
    """
    path = Path(file_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding=encoding, newline="") as f:
        if not data:
            return

        # Check if data is list of dicts
        if isinstance(data[0], dict):
            if fieldnames is None:
                fieldnames = list(data[0].keys())

            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
            if write_header:
                writer.writeheader()
            writer.writerows(data)
        else:
            # Data is list of lists
            writer = csv.writer(f, delimiter=delimiter)
            if write_header and fieldnames:
                writer.writerow(fieldnames)
            writer.writerows(data)


def read_csv_pandas(
    file_path: PathLike,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> Any:  # Returns pd.DataFrame if pandas is available
    """
    Read CSV file using pandas (if available).

    Args:
        file_path: Path to the CSV file
        encoding: File encoding (default: utf-8)
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        pandas DataFrame

    Raises:
        ImportError: If pandas is not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> df = read_csv_pandas("data.csv")
        >>> df = read_csv_pandas("data.csv", parse_dates=["date"])
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for this function")

    return pd.read_csv(file_path, encoding=encoding, **kwargs)


def write_csv_pandas(
    df: Any,  # pd.DataFrame
    file_path: PathLike,
    encoding: str = "utf-8",
    index: bool = False,
    create_dirs: bool = True,
    **kwargs: Any,
) -> None:
    """
    Write pandas DataFrame to CSV file.

    Args:
        df: pandas DataFrame to write
        file_path: Path to the output file
        encoding: File encoding (default: utf-8)
        index: If True, write row index
        create_dirs: If True, create parent directories if they don't exist
        **kwargs: Additional arguments passed to df.to_csv()

    Raises:
        ImportError: If pandas is not installed

    Examples:
        >>> df.to_csv("output.csv")
        >>> write_csv_pandas(df, "output.csv", index=False)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required for this function")

    path = Path(file_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, encoding=encoding, index=index, **kwargs)


def load_data(
    file_path: PathLike,
    file_format: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Load data from file, automatically detecting format from extension.

    Supported formats: json, yaml, yml, csv

    Args:
        file_path: Path to the file
        file_format: Explicit format (if None, detect from extension)
        **kwargs: Additional arguments passed to format-specific reader

    Returns:
        Loaded data (type depends on file format)

    Raises:
        ValueError: If file format is unsupported or cannot be detected

    Examples:
        >>> data = load_data("config.json")
        >>> data = load_data("data.csv")
        >>> data = load_data("settings.yaml")
    """
    path = Path(file_path)

    if file_format is None:
        file_format = path.suffix.lstrip(".").lower()

    format_readers = {
        "json": read_json,
        "yaml": read_yaml,
        "yml": read_yaml,
        "csv": read_csv,
    }

    reader = format_readers.get(file_format)
    if reader is None:
        raise ValueError(
            f"Unsupported file format: {file_format}. "
            f"Supported formats: {', '.join(format_readers.keys())}"
        )

    return reader(path, **kwargs)


def save_data(
    data: Any,
    file_path: PathLike,
    file_format: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Save data to file, automatically detecting format from extension.

    Supported formats: json, yaml, yml, csv

    Args:
        data: Data to save
        file_path: Path to the output file
        file_format: Explicit format (if None, detect from extension)
        **kwargs: Additional arguments passed to format-specific writer

    Raises:
        ValueError: If file format is unsupported or cannot be detected

    Examples:
        >>> save_data({"key": "value"}, "config.json")
        >>> save_data([{"a": 1}], "data.csv")
        >>> save_data({"setting": True}, "config.yaml")
    """
    path = Path(file_path)

    if file_format is None:
        file_format = path.suffix.lstrip(".").lower()

    format_writers = {
        "json": write_json,
        "yaml": write_yaml,
        "yml": write_yaml,
        "csv": write_csv,
    }

    writer = format_writers.get(file_format)
    if writer is None:
        raise ValueError(
            f"Unsupported file format: {file_format}. "
            f"Supported formats: {', '.join(format_writers.keys())}"
        )

    writer(data, path, **kwargs)


def file_exists(file_path: PathLike) -> bool:
    """
    Check if a file exists.

    Args:
        file_path: Path to check

    Returns:
        True if file exists, False otherwise

    Examples:
        >>> file_exists("config.json")
        True
        >>> file_exists("missing.txt")
        False
    """
    return Path(file_path).is_file()


def ensure_directory(dir_path: PathLike) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Path to the directory

    Returns:
        Path object for the directory

    Examples:
        >>> ensure_directory("data/output")
        PosixPath('data/output')
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(file_path: PathLike) -> int:
    """
    Get the size of a file in bytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> size = get_file_size("data.json")
        >>> print(f"Size: {size} bytes")
    """
    return Path(file_path).stat().st_size


def list_files(
    directory: PathLike,
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*" for all files)
        recursive: If True, search recursively

    Returns:
        List of Path objects for matching files

    Examples:
        >>> files = list_files("data", "*.json")
        >>> files = list_files("data", "**/*.csv", recursive=True)
    """
    path = Path(directory)

    if recursive:
        return [f for f in path.rglob(pattern) if f.is_file()]
    else:
        return [f for f in path.glob(pattern) if f.is_file()]


def backup_file(file_path: PathLike, backup_suffix: str = ".bak") -> Path:
    """
    Create a backup copy of a file.

    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix to append to backup filename

    Returns:
        Path to the backup file

    Raises:
        FileNotFoundError: If source file doesn't exist

    Examples:
        >>> backup_path = backup_file("important.json")
        >>> backup_path = backup_file("data.csv", ".backup")
    """
    import shutil

    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    backup = source.with_suffix(source.suffix + backup_suffix)
    shutil.copy2(source, backup)
    return backup
