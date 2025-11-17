"""
Data loading utilities for NOCT and module specifications.

This module provides functions to load and manage B03 NOCT data and other
module specifications from various data sources.
"""

from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime

from pv_simulator.models.noct import ModuleNOCTData, NOCTSpecification, NOCTTestConditions


class NOCTDataLoader:
    """
    Loader for NOCT data from CSV files and databases.

    This class handles loading, caching, and querying of module NOCT data,
    particularly for B03-verified modules.

    Attributes:
        data_path: Path to the NOCT data CSV file
        _cache: Cached DataFrame of NOCT data
        _modules_cache: Dictionary cache of ModuleNOCTData objects

    Examples:
        >>> loader = NOCTDataLoader()
        >>> loader.load_data()
        >>> module = loader.get_module_by_id("B03-00001")
        >>> module.manufacturer
        'SunPower'
    """

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the NOCT data loader.

        Args:
            data_path: Path to NOCT data CSV file. If None, uses default B03 data path.
        """
        if data_path is None:
            # Default to B03 NOCT data
            default_path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "noct" / "b03_noct_data.csv"
            self.data_path = default_path
        else:
            self.data_path = Path(data_path)

        self._cache: Optional[pd.DataFrame] = None
        self._modules_cache: Dict[str, ModuleNOCTData] = {}

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load NOCT data from CSV file.

        Args:
            force_reload: If True, reload data even if cached

        Returns:
            DataFrame containing NOCT data

        Raises:
            FileNotFoundError: If data file does not exist
            ValueError: If data file is malformed

        Examples:
            >>> loader = NOCTDataLoader()
            >>> df = loader.load_data()
            >>> 'module_id' in df.columns
            True
        """
        if self._cache is not None and not force_reload:
            return self._cache

        if not self.data_path.exists():
            raise FileNotFoundError(f"NOCT data file not found: {self.data_path}")

        try:
            df = pd.read_csv(self.data_path)

            # Validate required columns
            required_columns = [
                "module_id",
                "manufacturer",
                "model_name",
                "technology",
                "noct_celsius",
                "temp_coeff_power",
                "temp_coeff_voc",
                "rated_power_stc",
                "efficiency_stc",
                "module_area",
            ]

            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            self._cache = df
            return df

        except Exception as e:
            raise ValueError(f"Error loading NOCT data: {e}")

    def get_module_by_id(self, module_id: str) -> Optional[ModuleNOCTData]:
        """
        Get module NOCT data by module ID.

        Args:
            module_id: Module identifier (e.g., "B03-00001")

        Returns:
            ModuleNOCTData object or None if not found

        Examples:
            >>> loader = NOCTDataLoader()
            >>> loader.load_data()
            >>> module = loader.get_module_by_id("B03-00001")
            >>> module.manufacturer
            'SunPower'
        """
        # Check cache first
        if module_id in self._modules_cache:
            return self._modules_cache[module_id]

        # Load data if not already loaded
        if self._cache is None:
            self.load_data()

        # Find module in dataframe
        module_row = self._cache[self._cache["module_id"] == module_id]

        if module_row.empty:
            return None

        # Convert to ModuleNOCTData
        row = module_row.iloc[0]
        module_data = self._row_to_module_noct_data(row)

        # Cache the result
        self._modules_cache[module_id] = module_data

        return module_data

    def get_modules_by_manufacturer(self, manufacturer: str) -> List[ModuleNOCTData]:
        """
        Get all modules from a specific manufacturer.

        Args:
            manufacturer: Manufacturer name

        Returns:
            List of ModuleNOCTData objects

        Examples:
            >>> loader = NOCTDataLoader()
            >>> loader.load_data()
            >>> modules = loader.get_modules_by_manufacturer("SunPower")
            >>> len(modules) > 0
            True
        """
        if self._cache is None:
            self.load_data()

        manufacturer_rows = self._cache[
            self._cache["manufacturer"].str.contains(manufacturer, case=False, na=False)
        ]

        modules = []
        for _, row in manufacturer_rows.iterrows():
            modules.append(self._row_to_module_noct_data(row))

        return modules

    def get_modules_by_technology(self, technology: str) -> List[ModuleNOCTData]:
        """
        Get all modules of a specific technology type.

        Args:
            technology: Technology type (e.g., "mono_si", "bifacial", "hjt")

        Returns:
            List of ModuleNOCTData objects

        Examples:
            >>> loader = NOCTDataLoader()
            >>> loader.load_data()
            >>> modules = loader.get_modules_by_technology("mono_si")
            >>> len(modules) > 0
            True
        """
        if self._cache is None:
            self.load_data()

        tech_rows = self._cache[self._cache["technology"] == technology]

        modules = []
        for _, row in tech_rows.iterrows():
            modules.append(self._row_to_module_noct_data(row))

        return modules

    def get_b03_verified_modules(self) -> List[ModuleNOCTData]:
        """
        Get all B03-verified modules.

        Returns:
            List of B03-verified ModuleNOCTData objects

        Examples:
            >>> loader = NOCTDataLoader()
            >>> loader.load_data()
            >>> modules = loader.get_b03_verified_modules()
            >>> all(m.b03_verified for m in modules)
            True
        """
        if self._cache is None:
            self.load_data()

        b03_rows = self._cache[self._cache["b03_verified"] == True]

        modules = []
        for _, row in b03_rows.iterrows():
            modules.append(self._row_to_module_noct_data(row))

        return modules

    def _row_to_module_noct_data(self, row: pd.Series) -> ModuleNOCTData:
        """
        Convert a DataFrame row to ModuleNOCTData object.

        Args:
            row: DataFrame row containing module data

        Returns:
            ModuleNOCTData object
        """
        # Create NOCT specification
        noct_spec = NOCTSpecification(
            noct_celsius=float(row["noct_celsius"]),
            test_conditions=NOCTTestConditions(),
            test_date=pd.to_datetime(row.get("test_date", None)).date() if pd.notna(row.get("test_date")) else None,
            notes=str(row.get("notes", "")) if pd.notna(row.get("notes")) else None,
        )

        # Create ModuleNOCTData
        module_data = ModuleNOCTData(
            module_id=str(row["module_id"]),
            manufacturer=str(row["manufacturer"]),
            model_name=str(row["model_name"]),
            technology=str(row["technology"]),
            noct_spec=noct_spec,
            temp_coeff_power=float(row["temp_coeff_power"]),
            temp_coeff_voc=float(row["temp_coeff_voc"]),
            temp_coeff_isc=float(row.get("temp_coeff_isc", 0.0005)),
            rated_power_stc=float(row["rated_power_stc"]),
            efficiency_stc=float(row["efficiency_stc"]),
            module_area=float(row["module_area"]),
            cell_count=int(row.get("cell_count", 60)) if pd.notna(row.get("cell_count")) else 60,
            heat_capacity=float(row.get("heat_capacity", 11000.0)),
            absorptivity=float(row.get("absorptivity", 0.9)),
            emissivity=float(row.get("emissivity", 0.85)),
            b03_verified=bool(row.get("b03_verified", False)),
            data_source=str(row.get("data_source", "unknown")),
        )

        return module_data

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded NOCT database.

        Returns:
            Dictionary containing database statistics

        Examples:
            >>> loader = NOCTDataLoader()
            >>> loader.load_data()
            >>> stats = loader.get_statistics()
            >>> stats['total_modules'] > 0
            True
        """
        if self._cache is None:
            self.load_data()

        stats = {
            "total_modules": len(self._cache),
            "b03_verified": int(self._cache["b03_verified"].sum()),
            "manufacturers": self._cache["manufacturer"].nunique(),
            "technologies": self._cache["technology"].value_counts().to_dict(),
            "noct_range": {
                "min": float(self._cache["noct_celsius"].min()),
                "max": float(self._cache["noct_celsius"].max()),
                "mean": float(self._cache["noct_celsius"].mean()),
                "median": float(self._cache["noct_celsius"].median()),
            },
            "efficiency_range": {
                "min": float(self._cache["efficiency_stc"].min()),
                "max": float(self._cache["efficiency_stc"].max()),
                "mean": float(self._cache["efficiency_stc"].mean()),
            },
            "power_range": {
                "min": float(self._cache["rated_power_stc"].min()),
                "max": float(self._cache["rated_power_stc"].max()),
                "mean": float(self._cache["rated_power_stc"].mean()),
            },
        }

        return stats


def load_b03_noct_database(data_path: Optional[Path] = None) -> NOCTDataLoader:
    """
    Convenience function to load B03 NOCT database.

    Args:
        data_path: Optional custom path to NOCT data

    Returns:
        Loaded NOCTDataLoader instance

    Examples:
        >>> loader = load_b03_noct_database()
        >>> loader.load_data()
        >>> stats = loader.get_statistics()
        >>> stats['total_modules'] > 0
        True
    """
    loader = NOCTDataLoader(data_path=data_path)
    loader.load_data()
    return loader
