"""
Tests for file I/O utilities.
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.pv_simulator.utils.file_io import (
    read_json,
    write_json,
    read_yaml,
    write_yaml,
    read_csv,
    write_csv,
    load_data,
    save_data,
    file_exists,
    ensure_directory,
    get_file_size,
    list_files,
    backup_file,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestJSONOperations:
    """Tests for JSON read/write operations."""

    def test_write_and_read_json(self, temp_dir):
        """Test writing and reading JSON file."""
        file_path = temp_dir / "test.json"
        test_data = {"name": "Solar Panel", "power": 300, "efficiency": 0.18}

        write_json(test_data, file_path)
        result = read_json(file_path)

        assert result == test_data

    def test_write_json_with_lists(self, temp_dir):
        """Test JSON with list data."""
        file_path = temp_dir / "list.json"
        test_data = [1, 2, 3, 4, 5]

        write_json(test_data, file_path)
        result = read_json(file_path)

        assert result == test_data

    def test_read_json_missing_file_with_default(self, temp_dir):
        """Test reading missing file returns default."""
        file_path = temp_dir / "missing.json"
        result = read_json(file_path, default={})

        assert result == {}

    def test_read_json_missing_file_no_default(self, temp_dir):
        """Test reading missing file raises error."""
        file_path = temp_dir / "missing.json"

        with pytest.raises(FileNotFoundError):
            read_json(file_path)

    def test_write_json_creates_directories(self, temp_dir):
        """Test JSON write creates parent directories."""
        file_path = temp_dir / "subdir" / "deep" / "test.json"
        test_data = {"test": "value"}

        write_json(test_data, file_path, create_dirs=True)

        assert file_path.exists()
        result = read_json(file_path)
        assert result == test_data


class TestYAMLOperations:
    """Tests for YAML read/write operations."""

    def test_write_and_read_yaml(self, temp_dir):
        """Test writing and reading YAML file."""
        file_path = temp_dir / "test.yaml"
        test_data = {
            "system": {"name": "PV Array", "capacity": 100},
            "modules": [{"type": "mono", "count": 50}],
        }

        write_yaml(test_data, file_path)
        result = read_yaml(file_path)

        assert result == test_data

    def test_read_yaml_missing_file_with_default(self, temp_dir):
        """Test reading missing YAML file returns default."""
        file_path = temp_dir / "missing.yaml"
        result = read_yaml(file_path, default={})

        assert result == {}

    def test_write_yaml_creates_directories(self, temp_dir):
        """Test YAML write creates parent directories."""
        file_path = temp_dir / "config" / "settings.yaml"
        test_data = {"setting": "value"}

        write_yaml(test_data, file_path, create_dirs=True)

        assert file_path.exists()


class TestCSVOperations:
    """Tests for CSV read/write operations."""

    def test_write_and_read_csv_dict(self, temp_dir):
        """Test writing and reading CSV with dictionaries."""
        file_path = temp_dir / "data.csv"
        test_data = [
            {"name": "Module1", "power": 300, "efficiency": 18.5},
            {"name": "Module2", "power": 350, "efficiency": 20.0},
        ]

        write_csv(test_data, file_path)
        result = read_csv(file_path)

        assert len(result) == 2
        assert result[0]["name"] == "Module1"
        assert result[1]["power"] == "350"  # CSV reads as strings

    def test_write_csv_with_custom_delimiter(self, temp_dir):
        """Test CSV with custom delimiter."""
        file_path = temp_dir / "data.tsv"
        test_data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        write_csv(test_data, file_path, delimiter="\t")
        result = read_csv(file_path, delimiter="\t")

        assert len(result) == 2

    def test_write_csv_list_of_lists(self, temp_dir):
        """Test writing list of lists."""
        file_path = temp_dir / "matrix.csv"
        test_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        write_csv(test_data, file_path, fieldnames=["A", "B", "C"])
        result = read_csv(file_path)

        assert len(result) == 3
        assert result[0]["A"] == "1"

    def test_write_empty_csv(self, temp_dir):
        """Test writing empty CSV."""
        file_path = temp_dir / "empty.csv"
        write_csv([], file_path)

        assert file_path.exists()
        # Empty file should still be created


class TestGenericLoadSave:
    """Tests for generic load/save functions."""

    def test_load_json_by_extension(self, temp_dir):
        """Test load detects JSON by extension."""
        file_path = temp_dir / "data.json"
        test_data = {"key": "value"}

        write_json(test_data, file_path)
        result = load_data(file_path)

        assert result == test_data

    def test_load_yaml_by_extension(self, temp_dir):
        """Test load detects YAML by extension."""
        file_path = temp_dir / "config.yaml"
        test_data = {"config": "setting"}

        write_yaml(test_data, file_path)
        result = load_data(file_path)

        assert result == test_data

    def test_load_yml_extension(self, temp_dir):
        """Test load handles .yml extension."""
        file_path = temp_dir / "config.yml"
        test_data = {"config": "setting"}

        write_yaml(test_data, file_path)
        result = load_data(file_path)

        assert result == test_data

    def test_save_json_by_extension(self, temp_dir):
        """Test save detects JSON by extension."""
        file_path = temp_dir / "output.json"
        test_data = {"result": 123}

        save_data(test_data, file_path)

        assert file_path.exists()
        with open(file_path) as f:
            result = json.load(f)
        assert result == test_data

    def test_save_explicit_format(self, temp_dir):
        """Test save with explicit format."""
        file_path = temp_dir / "data.txt"
        test_data = {"key": "value"}

        save_data(test_data, file_path, file_format="json")

        result = read_json(file_path)
        assert result == test_data

    def test_load_unsupported_format(self, temp_dir):
        """Test load with unsupported format raises error."""
        file_path = temp_dir / "data.xyz"

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(file_path)


class TestFileUtilities:
    """Tests for file utility functions."""

    def test_file_exists(self, temp_dir):
        """Test file existence check."""
        file_path = temp_dir / "test.txt"

        assert not file_exists(file_path)

        file_path.write_text("test")

        assert file_exists(file_path)

    def test_ensure_directory(self, temp_dir):
        """Test directory creation."""
        dir_path = temp_dir / "new" / "nested" / "directory"

        result = ensure_directory(dir_path)

        assert result.exists()
        assert result.is_dir()

    def test_ensure_directory_existing(self, temp_dir):
        """Test ensure_directory on existing directory."""
        dir_path = temp_dir / "existing"
        dir_path.mkdir()

        result = ensure_directory(dir_path)

        assert result.exists()

    def test_get_file_size(self, temp_dir):
        """Test getting file size."""
        file_path = temp_dir / "test.txt"
        content = "Hello, World!"
        file_path.write_text(content)

        size = get_file_size(file_path)

        assert size == len(content.encode())

    def test_get_file_size_missing(self, temp_dir):
        """Test getting size of missing file raises error."""
        file_path = temp_dir / "missing.txt"

        with pytest.raises(FileNotFoundError):
            get_file_size(file_path)

    def test_list_files(self, temp_dir):
        """Test listing files in directory."""
        # Create test files
        (temp_dir / "file1.txt").write_text("test")
        (temp_dir / "file2.txt").write_text("test")
        (temp_dir / "data.json").write_text("{}")

        # List all files
        files = list_files(temp_dir)
        assert len(files) == 3

        # List txt files only
        txt_files = list_files(temp_dir, "*.txt")
        assert len(txt_files) == 2

    def test_list_files_recursive(self, temp_dir):
        """Test recursive file listing."""
        # Create nested structure
        (temp_dir / "file1.txt").write_text("test")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("test")

        # Non-recursive
        files = list_files(temp_dir, "*.txt", recursive=False)
        assert len(files) == 1

        # Recursive
        files = list_files(temp_dir, "**/*.txt", recursive=True)
        assert len(files) == 2

    def test_backup_file(self, temp_dir):
        """Test file backup creation."""
        original = temp_dir / "important.txt"
        original.write_text("important data")

        backup_path = backup_file(original)

        assert backup_path.exists()
        assert backup_path.name == "important.txt.bak"
        assert backup_path.read_text() == "important data"

    def test_backup_file_custom_suffix(self, temp_dir):
        """Test backup with custom suffix."""
        original = temp_dir / "data.json"
        original.write_text("{}")

        backup_path = backup_file(original, backup_suffix=".backup")

        assert backup_path.exists()
        assert backup_path.name == "data.json.backup"

    def test_backup_missing_file(self, temp_dir):
        """Test backup of missing file raises error."""
        missing = temp_dir / "missing.txt"

        with pytest.raises(FileNotFoundError):
            backup_file(missing)
