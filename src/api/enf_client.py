"""
ENF Solar API Client for PV Circularity Simulator.

This module provides an interface to ENF Solar's database for supplier
information, ratings, and market data for photovoltaic materials and components.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import requests
from enum import Enum
import time


class SupplierType(Enum):
    """ENF Solar supplier types."""
    CELL_MANUFACTURER = "Cell Manufacturer"
    MODULE_MANUFACTURER = "Module Manufacturer"
    MATERIALS_SUPPLIER = "Materials Supplier"
    EQUIPMENT_SUPPLIER = "Equipment Supplier"
    WAFER_PRODUCER = "Wafer Producer"
    INGOT_PRODUCER = "Ingot Producer"


class ProductCategory(Enum):
    """Product categories in ENF Solar."""
    SILICON_WAFERS = "Silicon Wafers"
    SOLAR_CELLS = "Solar Cells"
    ENCAPSULANTS = "Encapsulants"
    BACKSHEETS = "Backsheets"
    GLASS = "Glass"
    FRAMES = "Frames"
    JUNCTION_BOXES = "Junction Boxes"
    CONDUCTIVE_PASTE = "Conductive Paste"
    RIBBONS = "Ribbons"


@dataclass
class ENFSupplier:
    """ENF Solar supplier information."""
    company_id: str
    company_name: str
    country: str
    supplier_type: SupplierType
    enf_tier: str  # Tier 1, Tier 2, Tier 3
    rating: float  # 0-5
    total_reviews: int
    verified: bool
    products: List[str]
    certifications: List[str]
    production_capacity: Optional[str] = None
    year_established: Optional[int] = None
    number_of_employees: Optional[str] = None
    contact_email: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    recent_projects: List[str] = None


@dataclass
class PriceQuote:
    """Price quote from supplier."""
    supplier_id: str
    supplier_name: str
    product_name: str
    unit_price: float
    currency: str
    minimum_order: str
    lead_time_days: int
    valid_until: str
    terms: str
    last_updated: str


@dataclass
class MarketPrice:
    """Market price data for materials."""
    material_name: str
    average_price: float
    min_price: float
    max_price: float
    currency: str
    unit: str
    date: str
    trend: str  # Rising/Stable/Falling
    volatility: str  # Low/Medium/High


class ENFAPIClient:
    """
    Client for interacting with ENF Solar API.

    Note: This is a mock implementation for demonstration.
    In production, this would connect to actual ENF Solar API endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.enfsolar.com/v1",
        cache_duration_hours: int = 24
    ):
        """
        Initialize ENF API client.

        Args:
            api_key: API key for ENF Solar (optional for demo)
            base_url: Base URL for API endpoints
            cache_duration_hours: How long to cache results
        """
        self.api_key = api_key
        self.base_url = base_url
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        # Mock mode when no API key provided
        self.mock_mode = api_key is None

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'PV-Circularity-Simulator/1.0'
        }
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age < self.cache_duration

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache.get(cache_key)
        return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = datetime.now()

    def search_suppliers(
        self,
        product_category: Optional[ProductCategory] = None,
        country: Optional[str] = None,
        min_rating: float = 0.0,
        tier: Optional[str] = None,
        verified_only: bool = False
    ) -> List[ENFSupplier]:
        """
        Search for suppliers in ENF Solar database.

        Args:
            product_category: Filter by product category
            country: Filter by country
            min_rating: Minimum ENF rating
            tier: Filter by tier (Tier 1, Tier 2, Tier 3)
            verified_only: Only return verified suppliers

        Returns:
            List of matching suppliers
        """
        # In mock mode, return sample data
        if self.mock_mode:
            return self._get_mock_suppliers(
                product_category, country, min_rating, tier, verified_only
            )

        # Build query parameters
        params = {}
        if product_category:
            params['category'] = product_category.value
        if country:
            params['country'] = country
        if min_rating:
            params['min_rating'] = min_rating
        if tier:
            params['tier'] = tier
        if verified_only:
            params['verified'] = 'true'

        # Check cache
        cache_key = f"suppliers_{str(params)}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            response = requests.get(
                f"{self.base_url}/suppliers",
                headers=self._get_headers(),
                params=params,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            suppliers = [self._parse_supplier(s) for s in data.get('suppliers', [])]

            # Cache results
            self._save_to_cache(cache_key, suppliers)

            return suppliers

        except requests.exceptions.RequestException as e:
            print(f"Error fetching suppliers: {e}")
            return []

    def get_supplier_details(self, supplier_id: str) -> Optional[ENFSupplier]:
        """
        Get detailed information about a specific supplier.

        Args:
            supplier_id: ENF Solar supplier ID

        Returns:
            Supplier details or None if not found
        """
        if self.mock_mode:
            return self._get_mock_supplier_details(supplier_id)

        cache_key = f"supplier_{supplier_id}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            response = requests.get(
                f"{self.base_url}/suppliers/{supplier_id}",
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()

            supplier = self._parse_supplier(response.json())
            self._save_to_cache(cache_key, supplier)

            return supplier

        except requests.exceptions.RequestException as e:
            print(f"Error fetching supplier details: {e}")
            return None

    def get_price_quotes(
        self,
        material_name: str,
        quantity_kg: Optional[float] = None
    ) -> List[PriceQuote]:
        """
        Get price quotes for a specific material.

        Args:
            material_name: Name of material
            quantity_kg: Quantity in kg (optional)

        Returns:
            List of price quotes from suppliers
        """
        if self.mock_mode:
            return self._get_mock_price_quotes(material_name, quantity_kg)

        params = {'material': material_name}
        if quantity_kg:
            params['quantity'] = quantity_kg

        cache_key = f"quotes_{str(params)}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            response = requests.get(
                f"{self.base_url}/quotes",
                headers=self._get_headers(),
                params=params,
                timeout=30
            )
            response.raise_for_status()

            quotes = [self._parse_quote(q) for q in response.json().get('quotes', [])]
            self._save_to_cache(cache_key, quotes)

            return quotes

        except requests.exceptions.RequestException as e:
            print(f"Error fetching price quotes: {e}")
            return []

    def get_market_prices(self, material_name: str) -> Optional[MarketPrice]:
        """
        Get current market price data for a material.

        Args:
            material_name: Name of material

        Returns:
            Market price data or None
        """
        if self.mock_mode:
            return self._get_mock_market_price(material_name)

        cache_key = f"market_{material_name}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached

        try:
            response = requests.get(
                f"{self.base_url}/market/prices",
                headers=self._get_headers(),
                params={'material': material_name},
                timeout=30
            )
            response.raise_for_status()

            price_data = self._parse_market_price(response.json())
            self._save_to_cache(cache_key, price_data)

            return price_data

        except requests.exceptions.RequestException as e:
            print(f"Error fetching market prices: {e}")
            return None

    def _parse_supplier(self, data: Dict) -> ENFSupplier:
        """Parse supplier data from API response."""
        return ENFSupplier(
            company_id=data['id'],
            company_name=data['name'],
            country=data['country'],
            supplier_type=SupplierType(data['type']),
            enf_tier=data.get('tier', 'Tier 2'),
            rating=data.get('rating', 0.0),
            total_reviews=data.get('total_reviews', 0),
            verified=data.get('verified', False),
            products=data.get('products', []),
            certifications=data.get('certifications', []),
            production_capacity=data.get('capacity'),
            year_established=data.get('year_established'),
            number_of_employees=data.get('employees'),
            contact_email=data.get('email'),
            website=data.get('website'),
            description=data.get('description'),
            recent_projects=data.get('projects', [])
        )

    def _parse_quote(self, data: Dict) -> PriceQuote:
        """Parse price quote from API response."""
        return PriceQuote(
            supplier_id=data['supplier_id'],
            supplier_name=data['supplier_name'],
            product_name=data['product'],
            unit_price=data['price'],
            currency=data.get('currency', 'USD'),
            minimum_order=data['moq'],
            lead_time_days=data['lead_time'],
            valid_until=data['valid_until'],
            terms=data.get('terms', ''),
            last_updated=data.get('updated', datetime.now().isoformat())
        )

    def _parse_market_price(self, data: Dict) -> MarketPrice:
        """Parse market price from API response."""
        return MarketPrice(
            material_name=data['material'],
            average_price=data['avg_price'],
            min_price=data['min_price'],
            max_price=data['max_price'],
            currency=data.get('currency', 'USD'),
            unit=data.get('unit', 'kg'),
            date=data.get('date', datetime.now().isoformat()[:10]),
            trend=data.get('trend', 'Stable'),
            volatility=data.get('volatility', 'Low')
        )

    # Mock data methods for demonstration
    def _get_mock_suppliers(
        self,
        product_category: Optional[ProductCategory],
        country: Optional[str],
        min_rating: float,
        tier: Optional[str],
        verified_only: bool
    ) -> List[ENFSupplier]:
        """Generate mock supplier data."""
        suppliers = [
            ENFSupplier(
                company_id="ENF-001",
                company_name="LONGi Solar",
                country="China",
                supplier_type=SupplierType.WAFER_PRODUCER,
                enf_tier="Tier 1",
                rating=4.8,
                total_reviews=156,
                verified=True,
                products=["Monocrystalline Wafers", "Solar Cells"],
                certifications=["ISO 9001", "ISO 14001", "ISO 45001"],
                production_capacity="45 GW",
                year_established=2000,
                number_of_employees="30,000+",
                website="www.longi-solar.com",
                description="World's leading solar wafer and module manufacturer"
            ),
            ENFSupplier(
                company_id="ENF-002",
                company_name="Wacker Chemie AG",
                country="Germany",
                supplier_type=SupplierType.MATERIALS_SUPPLIER,
                enf_tier="Tier 1",
                rating=4.7,
                total_reviews=89,
                verified=True,
                products=["Polysilicon", "Silicon Wafers"],
                certifications=["ISO 9001", "ISO 14001", "ISO 50001"],
                production_capacity="80,000 MT/year",
                year_established=1914,
                number_of_employees="14,000+",
                website="www.wacker.com",
                description="Premium polysilicon and specialty chemicals producer"
            ),
            ENFSupplier(
                company_id="ENF-003",
                company_name="Heraeus",
                country="Germany",
                supplier_type=SupplierType.MATERIALS_SUPPLIER,
                enf_tier="Tier 1",
                rating=4.9,
                total_reviews=124,
                verified=True,
                products=["Silver Paste", "Conductive Materials"],
                certifications=["ISO 9001", "ISO 14001"],
                production_capacity="Confidential",
                year_established=1851,
                number_of_employees="12,000+",
                website="www.heraeus.com",
                description="Leading precious metals and materials technology"
            ),
            ENFSupplier(
                company_id="ENF-004",
                company_name="Hanwha Solutions",
                country="South Korea",
                supplier_type=SupplierType.MATERIALS_SUPPLIER,
                enf_tier="Tier 1",
                rating=4.6,
                total_reviews=78,
                verified=True,
                products=["EVA Encapsulant", "Backsheets", "Sealants"],
                certifications=["ISO 9001", "UL", "IEC"],
                production_capacity="8.4 GW",
                year_established=1965,
                number_of_employees="7,000+",
                website="www.hanwha-solutions.com",
                description="Integrated solar materials and modules"
            ),
            ENFSupplier(
                company_id="ENF-005",
                company_name="Saint-Gobain",
                country="France",
                supplier_type=SupplierType.MATERIALS_SUPPLIER,
                enf_tier="Tier 1",
                rating=4.7,
                total_reviews=92,
                verified=True,
                products=["Low-Iron Glass", "Anti-Reflective Coating"],
                certifications=["ISO 9001", "ISO 14001"],
                production_capacity="8.5M m²/year",
                year_established=1665,
                number_of_employees="170,000+",
                website="www.saint-gobain.com",
                description="Global leader in sustainable building materials"
            )
        ]

        # Apply filters
        filtered = suppliers

        if min_rating > 0:
            filtered = [s for s in filtered if s.rating >= min_rating]

        if tier:
            filtered = [s for s in filtered if s.enf_tier == tier]

        if verified_only:
            filtered = [s for s in filtered if s.verified]

        if country:
            filtered = [s for s in filtered if s.country == country]

        return filtered

    def _get_mock_supplier_details(self, supplier_id: str) -> Optional[ENFSupplier]:
        """Get mock details for a supplier."""
        suppliers = self._get_mock_suppliers(None, None, 0, None, False)
        for supplier in suppliers:
            if supplier.company_id == supplier_id:
                return supplier
        return None

    def _get_mock_price_quotes(
        self,
        material_name: str,
        quantity_kg: Optional[float]
    ) -> List[PriceQuote]:
        """Generate mock price quotes."""
        base_prices = {
            "Silicon": 25.0,
            "Silver Paste": 850.0,
            "Aluminum": 3.2,
            "Copper": 12.5,
            "Glass": 1.8,
            "EVA": 6.5,
            "POE": 8.2,
            "PET": 4.5
        }

        # Find base price
        base_price = 10.0
        for key, price in base_prices.items():
            if key.lower() in material_name.lower():
                base_price = price
                break

        quotes = [
            PriceQuote(
                supplier_id="ENF-001",
                supplier_name="Premium Supplier A",
                product_name=material_name,
                unit_price=base_price * 1.05,
                currency="USD",
                minimum_order="100 kg",
                lead_time_days=30,
                valid_until=(datetime.now() + timedelta(days=30)).isoformat()[:10],
                terms="FOB, 30-day payment",
                last_updated=datetime.now().isoformat()
            ),
            PriceQuote(
                supplier_id="ENF-002",
                supplier_name="Standard Supplier B",
                product_name=material_name,
                unit_price=base_price,
                currency="USD",
                minimum_order="250 kg",
                lead_time_days=45,
                valid_until=(datetime.now() + timedelta(days=45)).isoformat()[:10],
                terms="CIF, 60-day payment",
                last_updated=datetime.now().isoformat()
            ),
            PriceQuote(
                supplier_id="ENF-003",
                supplier_name="Budget Supplier C",
                product_name=material_name,
                unit_price=base_price * 0.92,
                currency="USD",
                minimum_order="500 kg",
                lead_time_days=60,
                valid_until=(datetime.now() + timedelta(days=60)).isoformat()[:10],
                terms="EXW, prepayment",
                last_updated=datetime.now().isoformat()
            )
        ]

        return quotes

    def _get_mock_market_price(self, material_name: str) -> MarketPrice:
        """Generate mock market price data."""
        base_prices = {
            "Silicon": 25.0,
            "Silver": 850.0,
            "Aluminum": 3.2,
            "Copper": 12.5,
            "Glass": 1.8,
            "EVA": 6.5,
            "POE": 8.2,
            "PET": 4.5
        }

        avg_price = 10.0
        for key, price in base_prices.items():
            if key.lower() in material_name.lower():
                avg_price = price
                break

        return MarketPrice(
            material_name=material_name,
            average_price=avg_price,
            min_price=avg_price * 0.85,
            max_price=avg_price * 1.15,
            currency="USD",
            unit="kg",
            date=datetime.now().isoformat()[:10],
            trend="Stable",
            volatility="Low"
        )
