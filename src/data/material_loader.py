"""
Material Data Loader for PV Circularity Simulator.

This module provides functionality to load, manage, and query material data
for photovoltaic components including silicon, metals, polymers, glass, and
other materials used in solar panel manufacturing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class MaterialCategory(Enum):
    """Material category enumeration."""
    SILICON = "Silicon"
    METALS = "Metals"
    POLYMERS = "Polymers"
    GLASS = "Glass"
    ENCAPSULANTS = "Encapsulants"
    BACKSHEETS = "Backsheets"
    FRAMES = "Frames"
    JUNCTION_BOX = "Junction Box"
    ADHESIVES = "Adhesives"
    COATINGS = "Coatings"


class Standard(Enum):
    """Industry standards for materials."""
    IEC_61215 = "IEC 61215"
    IEC_61730 = "IEC 61730"
    ISO_9001 = "ISO 9001"
    ISO_14001 = "ISO 14001"
    UL_1703 = "UL 1703"
    ROHS = "RoHS"
    REACH = "REACH"


@dataclass
class MaterialProperties:
    """Detailed material properties."""
    # Physical properties
    density: float  # g/cm³
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    melting_point: Optional[float] = None  # °C

    # Electrical properties (if applicable)
    electrical_resistivity: Optional[float] = None  # Ω·m
    dielectric_strength: Optional[float] = None  # kV/mm

    # Mechanical properties
    tensile_strength: Optional[float] = None  # MPa
    elastic_modulus: Optional[float] = None  # GPa
    hardness: Optional[str] = None  # Mohs or other scale

    # Optical properties (for transparent materials)
    transmittance: Optional[float] = None  # %
    refractive_index: Optional[float] = None

    # Environmental resistance
    uv_resistance: Optional[str] = None  # Low/Medium/High
    moisture_resistance: Optional[str] = None  # Low/Medium/High
    temperature_coefficient: Optional[float] = None  # %/°C


@dataclass
class CircularityMetrics:
    """Circularity and sustainability metrics."""
    recyclability_score: float  # 0-100
    recycled_content: float  # % of recycled material
    carbon_footprint: float  # kg CO2e per kg
    embodied_energy: float  # MJ/kg
    water_footprint: float  # L/kg
    toxicity_rating: str  # Low/Medium/High
    end_of_life_recovery_rate: float  # %
    reusability_potential: str  # Low/Medium/High
    degradability: str  # Non-degradable/Slow/Medium/Fast


@dataclass
class SupplierInfo:
    """Supplier information."""
    name: str
    country: str
    enf_rating: Optional[float] = None  # ENF Solar rating
    certifications: List[str] = field(default_factory=list)
    annual_capacity: Optional[str] = None
    lead_time_days: Optional[int] = None
    minimum_order_quantity: Optional[str] = None
    contact_info: Optional[str] = None


@dataclass
class PriceHistory:
    """Price history data point."""
    date: str
    price_per_kg: float
    currency: str = "USD"
    source: str = ""


@dataclass
class Material:
    """Complete material specification."""
    # Basic information
    id: str
    name: str
    category: MaterialCategory
    subcategory: str
    description: str

    # Properties
    properties: MaterialProperties

    # Circularity
    circularity: CircularityMetrics

    # Commercial
    base_price_per_kg: float  # USD/kg
    price_history: List[PriceHistory] = field(default_factory=list)
    suppliers: List[SupplierInfo] = field(default_factory=list)

    # Quality and compliance
    standards_compliance: List[Standard] = field(default_factory=list)
    quality_grade: str = "Standard"  # Standard/Premium/Ultra-Premium

    # Performance for PV applications
    pv_efficiency_impact: Optional[float] = None  # % impact on module efficiency
    typical_thickness: Optional[float] = None  # mm
    lifetime_years: Optional[int] = None

    # Metadata
    data_source: str = ""
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    notes: str = ""


class MaterialLoader:
    """Load and manage material data."""

    def __init__(self, data_directory: Optional[Path] = None):
        """
        Initialize material loader.

        Args:
            data_directory: Path to directory containing material data files.
                          If None, uses default sample data.
        """
        self.data_directory = data_directory
        self._materials: Dict[str, Material] = {}
        self._load_materials()

    def _load_materials(self) -> None:
        """Load materials from data directory or create sample data."""
        if self.data_directory and self.data_directory.exists():
            self._load_from_files()
        else:
            self._create_sample_materials()

    def _load_from_files(self) -> None:
        """Load materials from JSON files in data directory."""
        json_files = list(self.data_directory.glob("*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    material = self._dict_to_material(data)
                    self._materials[material.id] = material
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    def _dict_to_material(self, data: Dict) -> Material:
        """Convert dictionary to Material object."""
        # Convert category string to enum
        category = MaterialCategory[data['category']]

        # Parse properties
        properties = MaterialProperties(**data['properties'])

        # Parse circularity metrics
        circularity = CircularityMetrics(**data['circularity'])

        # Parse suppliers
        suppliers = [SupplierInfo(**s) for s in data.get('suppliers', [])]

        # Parse price history
        price_history = [PriceHistory(**p) for p in data.get('price_history', [])]

        # Parse standards
        standards = [Standard[s] for s in data.get('standards_compliance', [])]

        return Material(
            id=data['id'],
            name=data['name'],
            category=category,
            subcategory=data['subcategory'],
            description=data['description'],
            properties=properties,
            circularity=circularity,
            base_price_per_kg=data['base_price_per_kg'],
            price_history=price_history,
            suppliers=suppliers,
            standards_compliance=standards,
            quality_grade=data.get('quality_grade', 'Standard'),
            pv_efficiency_impact=data.get('pv_efficiency_impact'),
            typical_thickness=data.get('typical_thickness'),
            lifetime_years=data.get('lifetime_years'),
            data_source=data.get('data_source', ''),
            last_updated=data.get('last_updated', datetime.now().isoformat()),
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )

    def _create_sample_materials(self) -> None:
        """Create comprehensive sample materials database."""

        # Silicon materials
        silicon_materials = [
            Material(
                id="SI-MONO-001",
                name="Monocrystalline Silicon Wafer",
                category=MaterialCategory.SILICON,
                subcategory="Monocrystalline",
                description="High-purity monocrystalline silicon wafer with superior efficiency",
                properties=MaterialProperties(
                    density=2.33,
                    thermal_conductivity=149.0,
                    specific_heat=700.0,
                    melting_point=1414.0,
                    electrical_resistivity=0.001,
                    tensile_strength=180.0,
                    elastic_modulus=190.0,
                    hardness="7 Mohs"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=85.0,
                    recycled_content=5.0,
                    carbon_footprint=45.0,
                    embodied_energy=850.0,
                    water_footprint=2500.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=75.0,
                    reusability_potential="Medium",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=25.0,
                price_history=[
                    PriceHistory("2024-01", 28.5, "USD", "Market Report"),
                    PriceHistory("2024-06", 26.2, "USD", "Market Report"),
                    PriceHistory("2024-12", 25.0, "USD", "Market Report")
                ],
                suppliers=[
                    SupplierInfo(
                        name="LONGi Solar",
                        country="China",
                        enf_rating=4.8,
                        certifications=["ISO 9001", "ISO 14001"],
                        annual_capacity="45 GW",
                        lead_time_days=30,
                        minimum_order_quantity="100 kg"
                    ),
                    SupplierInfo(
                        name="Wacker Chemie",
                        country="Germany",
                        enf_rating=4.7,
                        certifications=["ISO 9001", "ISO 14001", "ISO 50001"],
                        annual_capacity="80,000 MT",
                        lead_time_days=45
                    )
                ],
                standards_compliance=[Standard.IEC_61215, Standard.ISO_9001, Standard.ISO_14001],
                quality_grade="Premium",
                pv_efficiency_impact=22.5,
                typical_thickness=0.18,
                lifetime_years=25,
                data_source="Industry Database",
                tags=["high-efficiency", "premium", "c-Si"]
            ),
            Material(
                id="SI-POLY-001",
                name="Polycrystalline Silicon Wafer",
                category=MaterialCategory.SILICON,
                subcategory="Polycrystalline",
                description="Cost-effective polycrystalline silicon with good performance",
                properties=MaterialProperties(
                    density=2.33,
                    thermal_conductivity=148.0,
                    specific_heat=700.0,
                    melting_point=1414.0,
                    electrical_resistivity=0.002,
                    tensile_strength=165.0,
                    elastic_modulus=185.0,
                    hardness="7 Mohs"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=80.0,
                    recycled_content=8.0,
                    carbon_footprint=38.0,
                    embodied_energy=720.0,
                    water_footprint=2200.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=72.0,
                    reusability_potential="Medium",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=18.5,
                standards_compliance=[Standard.IEC_61215, Standard.ISO_9001],
                quality_grade="Standard",
                pv_efficiency_impact=18.5,
                typical_thickness=0.20,
                lifetime_years=25,
                tags=["cost-effective", "mc-Si"]
            )
        ]

        # Metal materials
        metal_materials = [
            Material(
                id="MT-AG-001",
                name="Silver Paste (Front Contact)",
                category=MaterialCategory.METALS,
                subcategory="Conductive Paste",
                description="High-conductivity silver paste for front contact metallization",
                properties=MaterialProperties(
                    density=10.49,
                    thermal_conductivity=429.0,
                    specific_heat=235.0,
                    melting_point=961.8,
                    electrical_resistivity=1.59e-8
                ),
                circularity=CircularityMetrics(
                    recyclability_score=95.0,
                    recycled_content=25.0,
                    carbon_footprint=1500.0,
                    embodied_energy=5400.0,
                    water_footprint=12000.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=90.0,
                    reusability_potential="High",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=850.0,
                price_history=[
                    PriceHistory("2024-01", 920.0, "USD"),
                    PriceHistory("2024-06", 880.0, "USD"),
                    PriceHistory("2024-12", 850.0, "USD")
                ],
                suppliers=[
                    SupplierInfo(
                        name="Heraeus",
                        country="Germany",
                        enf_rating=4.9,
                        certifications=["ISO 9001", "ISO 14001"],
                        lead_time_days=60
                    )
                ],
                standards_compliance=[Standard.ISO_9001, Standard.ROHS],
                quality_grade="Premium",
                pv_efficiency_impact=0.5,
                lifetime_years=30,
                tags=["high-cost", "precious-metal", "recyclable"]
            ),
            Material(
                id="MT-AL-001",
                name="Aluminum Frame",
                category=MaterialCategory.METALS,
                subcategory="Structural",
                description="Anodized aluminum alloy frame for module mounting",
                properties=MaterialProperties(
                    density=2.70,
                    thermal_conductivity=205.0,
                    specific_heat=900.0,
                    melting_point=660.0,
                    tensile_strength=310.0,
                    elastic_modulus=69.0,
                    hardness="2.75 Mohs"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=98.0,
                    recycled_content=35.0,
                    carbon_footprint=8.5,
                    embodied_energy=170.0,
                    water_footprint=450.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=95.0,
                    reusability_potential="High",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=3.2,
                price_history=[
                    PriceHistory("2024-01", 3.5, "USD"),
                    PriceHistory("2024-06", 3.3, "USD"),
                    PriceHistory("2024-12", 3.2, "USD")
                ],
                suppliers=[
                    SupplierInfo(
                        name="Constellium",
                        country="France",
                        enf_rating=4.6,
                        certifications=["ISO 9001", "ISO 14001", "ASI"],
                        annual_capacity="2.3M tonnes",
                        lead_time_days=45
                    )
                ],
                standards_compliance=[Standard.ISO_9001, Standard.ISO_14001],
                quality_grade="Standard",
                lifetime_years=30,
                tags=["structural", "highly-recyclable", "cost-effective"]
            ),
            Material(
                id="MT-CU-001",
                name="Copper Ribbon (Interconnect)",
                category=MaterialCategory.METALS,
                subcategory="Interconnect",
                description="Tin-coated copper ribbon for cell interconnection",
                properties=MaterialProperties(
                    density=8.96,
                    thermal_conductivity=401.0,
                    specific_heat=385.0,
                    melting_point=1085.0,
                    electrical_resistivity=1.68e-8,
                    tensile_strength=220.0
                ),
                circularity=CircularityMetrics(
                    recyclability_score=92.0,
                    recycled_content=40.0,
                    carbon_footprint=3.8,
                    embodied_energy=65.0,
                    water_footprint=380.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=88.0,
                    reusability_potential="High",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=12.5,
                standards_compliance=[Standard.ROHS, Standard.REACH],
                quality_grade="Premium",
                lifetime_years=25,
                tags=["interconnect", "conductive", "recyclable"]
            )
        ]

        # Glass materials
        glass_materials = [
            Material(
                id="GL-LI-001",
                name="Low-Iron Tempered Glass",
                category=MaterialCategory.GLASS,
                subcategory="Front Cover",
                description="High-transmittance low-iron tempered glass for front cover",
                properties=MaterialProperties(
                    density=2.50,
                    thermal_conductivity=1.05,
                    specific_heat=840.0,
                    melting_point=1400.0,
                    tensile_strength=120.0,
                    elastic_modulus=73.0,
                    hardness="5.5 Mohs",
                    transmittance=91.5,
                    refractive_index=1.52
                ),
                circularity=CircularityMetrics(
                    recyclability_score=75.0,
                    recycled_content=20.0,
                    carbon_footprint=0.85,
                    embodied_energy=15.5,
                    water_footprint=125.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=70.0,
                    reusability_potential="Medium",
                    degradability="Non-degradable"
                ),
                base_price_per_kg=1.8,
                price_history=[
                    PriceHistory("2024-01", 2.0, "USD"),
                    PriceHistory("2024-06", 1.9, "USD"),
                    PriceHistory("2024-12", 1.8, "USD")
                ],
                suppliers=[
                    SupplierInfo(
                        name="Saint-Gobain",
                        country="France",
                        enf_rating=4.7,
                        certifications=["ISO 9001", "ISO 14001"],
                        annual_capacity="8.5M m²",
                        lead_time_days=30
                    ),
                    SupplierInfo(
                        name="Flat Glass Group",
                        country="China",
                        enf_rating=4.5,
                        certifications=["ISO 9001"],
                        lead_time_days=45
                    )
                ],
                standards_compliance=[Standard.IEC_61215, Standard.IEC_61730],
                quality_grade="Premium",
                pv_efficiency_impact=1.5,
                typical_thickness=3.2,
                lifetime_years=30,
                tags=["high-transmittance", "anti-reflective"]
            )
        ]

        # Polymer materials
        polymer_materials = [
            Material(
                id="PM-EVA-001",
                name="EVA Encapsulant Film",
                category=MaterialCategory.POLYMERS,
                subcategory="Encapsulant",
                description="Ethylene-vinyl acetate copolymer for cell encapsulation",
                properties=MaterialProperties(
                    density=0.95,
                    thermal_conductivity=0.34,
                    specific_heat=2090.0,
                    melting_point=75.0,
                    dielectric_strength=40.0,
                    transmittance=90.0,
                    uv_resistance="High",
                    moisture_resistance="Medium"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=45.0,
                    recycled_content=0.0,
                    carbon_footprint=2.8,
                    embodied_energy=95.0,
                    water_footprint=280.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=25.0,
                    reusability_potential="Low",
                    degradability="Slow"
                ),
                base_price_per_kg=6.5,
                price_history=[
                    PriceHistory("2024-01", 7.2, "USD"),
                    PriceHistory("2024-06", 6.8, "USD"),
                    PriceHistory("2024-12", 6.5, "USD")
                ],
                suppliers=[
                    SupplierInfo(
                        name="Hanwha Solutions",
                        country="South Korea",
                        enf_rating=4.6,
                        certifications=["ISO 9001", "UL"],
                        lead_time_days=40
                    )
                ],
                standards_compliance=[Standard.IEC_61215, Standard.UL_1703],
                quality_grade="Premium",
                typical_thickness=0.45,
                lifetime_years=25,
                tags=["encapsulant", "industry-standard"]
            ),
            Material(
                id="PM-POE-001",
                name="POE Encapsulant Film",
                category=MaterialCategory.POLYMERS,
                subcategory="Encapsulant",
                description="Polyolefin elastomer with superior moisture resistance",
                properties=MaterialProperties(
                    density=0.87,
                    thermal_conductivity=0.24,
                    specific_heat=2200.0,
                    melting_point=65.0,
                    dielectric_strength=45.0,
                    transmittance=92.0,
                    uv_resistance="High",
                    moisture_resistance="High"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=50.0,
                    recycled_content=0.0,
                    carbon_footprint=2.5,
                    embodied_energy=88.0,
                    water_footprint=250.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=30.0,
                    reusability_potential="Low",
                    degradability="Slow"
                ),
                base_price_per_kg=8.2,
                suppliers=[
                    SupplierInfo(
                        name="Mitsui Chemicals",
                        country="Japan",
                        enf_rating=4.7,
                        certifications=["ISO 9001", "ISO 14001"],
                        lead_time_days=50
                    )
                ],
                standards_compliance=[Standard.IEC_61215, Standard.IEC_61730],
                quality_grade="Premium",
                pv_efficiency_impact=0.3,
                typical_thickness=0.45,
                lifetime_years=30,
                tags=["advanced", "moisture-resistant", "bifacial"]
            ),
            Material(
                id="PM-PET-001",
                name="PET Backsheet",
                category=MaterialCategory.POLYMERS,
                subcategory="Backsheet",
                description="Polyethylene terephthalate backsheet with excellent durability",
                properties=MaterialProperties(
                    density=1.38,
                    thermal_conductivity=0.15,
                    specific_heat=1200.0,
                    melting_point=260.0,
                    dielectric_strength=20.0,
                    tensile_strength=55.0,
                    uv_resistance="High",
                    moisture_resistance="High"
                ),
                circularity=CircularityMetrics(
                    recyclability_score=70.0,
                    recycled_content=15.0,
                    carbon_footprint=3.2,
                    embodied_energy=78.0,
                    water_footprint=210.0,
                    toxicity_rating="Low",
                    end_of_life_recovery_rate=55.0,
                    reusability_potential="Medium",
                    degradability="Slow"
                ),
                base_price_per_kg=4.5,
                standards_compliance=[Standard.IEC_61215, Standard.IEC_61730],
                quality_grade="Standard",
                typical_thickness=0.35,
                lifetime_years=25,
                tags=["backsheet", "durable", "recyclable"]
            )
        ]

        # Combine all materials
        all_materials = (
            silicon_materials + metal_materials +
            glass_materials + polymer_materials
        )

        for material in all_materials:
            self._materials[material.id] = material

    def get_material(self, material_id: str) -> Optional[Material]:
        """Get material by ID."""
        return self._materials.get(material_id)

    def get_all_materials(self) -> List[Material]:
        """Get all materials."""
        return list(self._materials.values())

    def get_by_category(self, category: MaterialCategory) -> List[Material]:
        """Get materials by category."""
        return [m for m in self._materials.values() if m.category == category]

    def search_materials(
        self,
        query: str = "",
        categories: Optional[Set[MaterialCategory]] = None,
        min_circularity: Optional[float] = None,
        max_price: Optional[float] = None,
        min_efficiency: Optional[float] = None,
        tags: Optional[Set[str]] = None
    ) -> List[Material]:
        """
        Search materials with filters.

        Args:
            query: Text search query for name/description
            categories: Filter by material categories
            min_circularity: Minimum recyclability score
            max_price: Maximum price per kg
            min_efficiency: Minimum PV efficiency impact
            tags: Filter by tags

        Returns:
            List of matching materials
        """
        results = list(self._materials.values())

        # Text search
        if query:
            query_lower = query.lower()
            results = [
                m for m in results
                if query_lower in m.name.lower() or
                   query_lower in m.description.lower() or
                   query_lower in m.subcategory.lower()
            ]

        # Category filter
        if categories:
            results = [m for m in results if m.category in categories]

        # Circularity filter
        if min_circularity is not None:
            results = [
                m for m in results
                if m.circularity.recyclability_score >= min_circularity
            ]

        # Price filter
        if max_price is not None:
            results = [m for m in results if m.base_price_per_kg <= max_price]

        # Efficiency filter
        if min_efficiency is not None:
            results = [
                m for m in results
                if m.pv_efficiency_impact is not None and
                   m.pv_efficiency_impact >= min_efficiency
            ]

        # Tags filter
        if tags:
            results = [
                m for m in results
                if any(tag in m.tags for tag in tags)
            ]

        return results

    def get_materials_dataframe(
        self,
        materials: Optional[List[Material]] = None
    ) -> pd.DataFrame:
        """
        Convert materials to pandas DataFrame for analysis.

        Args:
            materials: List of materials to convert, or None for all

        Returns:
            DataFrame with material data
        """
        if materials is None:
            materials = self.get_all_materials()

        data = []
        for m in materials:
            data.append({
                'ID': m.id,
                'Name': m.name,
                'Category': m.category.value,
                'Subcategory': m.subcategory,
                'Price (USD/kg)': m.base_price_per_kg,
                'Recyclability': m.circularity.recyclability_score,
                'Carbon Footprint': m.circularity.carbon_footprint,
                'PV Efficiency Impact': m.pv_efficiency_impact or 0,
                'Quality': m.quality_grade,
                'Lifetime (years)': m.lifetime_years or 0,
                'Suppliers': len(m.suppliers)
            })

        return pd.DataFrame(data)

    def get_comparison_data(
        self,
        material_ids: List[str]
    ) -> Tuple[List[Material], pd.DataFrame]:
        """
        Get materials and comparison DataFrame for given IDs.

        Args:
            material_ids: List of material IDs to compare

        Returns:
            Tuple of (materials list, comparison DataFrame)
        """
        materials = [self.get_material(mid) for mid in material_ids]
        materials = [m for m in materials if m is not None]

        if not materials:
            return [], pd.DataFrame()

        df = self.get_materials_dataframe(materials)
        return materials, df
