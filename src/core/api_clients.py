"""
ENF Solar API Client
Provides integration with ENF Solar's component database for PV materials and components.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from functools import lru_cache, wraps
from datetime import datetime, timedelta
from collections import deque
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass


class APIError(Exception):
    """Raised when API returns an error"""
    pass


class RateLimiter:
    """
    Rate limiter using sliding window algorithm
    Limits requests to max_requests per time_window
    """

    def __init__(self, max_requests: int = 100, time_window: int = 3600):
        """
        Initialize rate limiter

        Args:
            max_requests: Maximum number of requests allowed (default: 100)
            time_window: Time window in seconds (default: 3600 = 1 hour)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()

    def __call__(self, func):
        """Decorator to apply rate limiting to functions"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self._check_rate_limit()
            result = func(*args, **kwargs)
            self._record_request()
            return result
        return wrapper

    def _check_rate_limit(self):
        """Check if rate limit would be exceeded"""
        now = datetime.now()
        # Remove requests outside the time window
        while self.requests and (now - self.requests[0]) > timedelta(seconds=self.time_window):
            self.requests.popleft()

        if len(self.requests) >= self.max_requests:
            oldest_request = self.requests[0]
            wait_time = (oldest_request + timedelta(seconds=self.time_window) - now).total_seconds()
            raise RateLimitError(
                f"Rate limit exceeded. Maximum {self.max_requests} requests per hour. "
                f"Please wait {wait_time:.0f} seconds."
            )

    def _record_request(self):
        """Record a new request timestamp"""
        self.requests.append(datetime.now())

    def reset(self):
        """Reset the rate limiter (useful for testing)"""
        self.requests.clear()


def retry_on_failure(max_retries: int = 3, backoff_factor: float = 2.0):
    """
    Decorator to retry API calls on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, APIError) as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = backoff_factor ** attempt
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Request failed after {max_retries + 1} attempts")

            raise last_exception
        return wrapper
    return decorator


class ENFSolarClient:
    """
    Client for interacting with the ENF Solar API

    Provides access to PV components database including modules, cells, inverters,
    mounting systems, cables, and connectors with comprehensive search and filtering.
    """

    # Supported component categories
    VALID_CATEGORIES = ['modules', 'cells', 'inverters', 'mounting', 'cables', 'connectors']

    # Valid filter keys
    VALID_FILTERS = ['power_range', 'efficiency_range', 'technology', 'manufacturer', 'price_range']

    def __init__(self, api_key: str):
        """
        Initialize ENF Solar API client

        Args:
            api_key: API key for authentication
        """
        if not api_key:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = 'https://www.enfsolar.com/api'
        self.rate_limiter = RateLimiter(max_requests=100, time_window=3600)

        # Configure session with connection pooling and retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Set default headers
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PV-Circularity-Simulator/1.0'
        })

        logger.info("ENFSolarClient initialized successfully")

    def _validate_category(self, category: str) -> None:
        """Validate component category"""
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{category}'. Must be one of: {', '.join(self.VALID_CATEGORIES)}"
            )

    def _validate_filters(self, filters: Dict[str, Any]) -> None:
        """Validate filter parameters"""
        if not isinstance(filters, dict):
            raise TypeError("Filters must be a dictionary")

        invalid_filters = set(filters.keys()) - set(self.VALID_FILTERS)
        if invalid_filters:
            logger.warning(f"Ignoring invalid filters: {', '.join(invalid_filters)}")

    @retry_on_failure(max_retries=3, backoff_factor=2.0)
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make HTTP request to API with rate limiting and error handling

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests

        Returns:
            Response data as dictionary

        Raises:
            RateLimitError: If rate limit is exceeded
            APIError: If API returns an error
        """
        # Apply rate limiting
        self.rate_limiter._check_rate_limit()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, timeout=30, **kwargs)

            # Record request for rate limiting
            self.rate_limiter._record_request()

            # Handle HTTP errors
            if response.status_code == 401:
                raise APIError("Authentication failed. Check your API key.")
            elif response.status_code == 403:
                raise APIError("Access forbidden. Check your API permissions.")
            elif response.status_code == 404:
                raise APIError(f"Endpoint not found: {endpoint}")
            elif response.status_code == 429:
                raise RateLimitError("API rate limit exceeded")
            elif response.status_code >= 400:
                raise APIError(f"API error {response.status_code}: {response.text}")

            response.raise_for_status()

            # Parse JSON response
            try:
                return response.json()
            except ValueError:
                raise APIError("Invalid JSON response from API")

        except requests.exceptions.Timeout:
            raise APIError("Request timeout. The API is not responding.")
        except requests.exceptions.ConnectionError:
            raise APIError("Connection error. Check your internet connection.")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")

    def search_components(self, category: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """
        Search for components by category and filters

        Args:
            category: Component category (modules, cells, inverters, mounting, cables, connectors)
            filters: Optional filters dictionary with keys:
                - power_range: tuple (min_watts, max_watts)
                - efficiency_range: tuple (min_percent, max_percent)
                - technology: str (e.g., 'monocrystalline', 'polycrystalline')
                - manufacturer: str
                - price_range: tuple (min_price, max_price)

        Returns:
            List of component dictionaries matching the search criteria

        Raises:
            ValueError: If category is invalid
            RateLimitError: If rate limit is exceeded
            APIError: If API request fails

        Example:
            >>> client = ENFSolarClient(api_key='your_key')
            >>> results = client.search_components(
            ...     'modules',
            ...     filters={'power_range': (300, 500), 'technology': 'monocrystalline'}
            ... )
        """
        self._validate_category(category)

        filters = filters or {}
        self._validate_filters(filters)

        # Build query parameters
        params = {'category': category}

        # Add filters to params
        if 'power_range' in filters and filters['power_range']:
            min_power, max_power = filters['power_range']
            params['min_power'] = min_power
            params['max_power'] = max_power

        if 'efficiency_range' in filters and filters['efficiency_range']:
            min_eff, max_eff = filters['efficiency_range']
            params['min_efficiency'] = min_eff
            params['max_efficiency'] = max_eff

        if 'technology' in filters:
            params['technology'] = filters['technology']

        if 'manufacturer' in filters:
            params['manufacturer'] = filters['manufacturer']

        if 'price_range' in filters and filters['price_range']:
            min_price, max_price = filters['price_range']
            params['min_price'] = min_price
            params['max_price'] = max_price

        logger.info(f"Searching {category} with filters: {filters}")

        response = self._make_request('GET', '/components/search', params=params)

        results = response.get('data', [])
        logger.info(f"Found {len(results)} components")

        return results

    def get_component_details(self, component_id: str) -> Dict:
        """
        Get detailed information about a specific component

        Args:
            component_id: Unique identifier for the component

        Returns:
            Dictionary with detailed component information including:
                - id: Component ID
                - name: Component name
                - manufacturer: Manufacturer name
                - category: Component category
                - specifications: Technical specifications
                - certifications: List of certifications
                - datasheet_url: URL to datasheet
                - last_updated: Last update timestamp

        Raises:
            ValueError: If component_id is empty
            RateLimitError: If rate limit is exceeded
            APIError: If API request fails or component not found

        Example:
            >>> details = client.get_component_details('MOD-12345')
        """
        if not component_id:
            raise ValueError("component_id is required")

        logger.info(f"Fetching details for component: {component_id}")

        response = self._make_request('GET', f'/components/{component_id}')

        component_data = response.get('data', {})

        if not component_data:
            raise APIError(f"Component not found: {component_id}")

        logger.info(f"Retrieved details for {component_data.get('name', 'Unknown')}")

        return component_data

    def get_pricing_data(self, component_id: str, region: str = 'global') -> Dict:
        """
        Get pricing information for a component in a specific region

        Args:
            component_id: Unique identifier for the component
            region: Geographic region (default: 'global')
                   Options: 'global', 'north_america', 'europe', 'asia', 'oceania', etc.

        Returns:
            Dictionary with pricing information:
                - component_id: Component ID
                - region: Region code
                - currency: Currency code (USD, EUR, etc.)
                - unit_price: Price per unit
                - bulk_pricing: List of bulk pricing tiers
                - last_updated: Last price update timestamp
                - availability: Stock availability status

        Raises:
            ValueError: If component_id is empty
            RateLimitError: If rate limit is exceeded
            APIError: If API request fails

        Example:
            >>> pricing = client.get_pricing_data('MOD-12345', region='europe')
        """
        if not component_id:
            raise ValueError("component_id is required")

        logger.info(f"Fetching pricing for component {component_id} in region: {region}")

        params = {'region': region}
        response = self._make_request('GET', f'/components/{component_id}/pricing', params=params)

        pricing_data = response.get('data', {})

        if not pricing_data:
            logger.warning(f"No pricing data available for component {component_id} in {region}")

        return pricing_data

    @lru_cache(maxsize=128)
    def list_manufacturers(self) -> List[str]:
        """
        Get list of all manufacturers in the database

        Results are cached to improve performance for repeated calls.

        Returns:
            Sorted list of manufacturer names

        Raises:
            RateLimitError: If rate limit is exceeded
            APIError: If API request fails

        Example:
            >>> manufacturers = client.list_manufacturers()
            >>> print(f"Found {len(manufacturers)} manufacturers")
        """
        logger.info("Fetching manufacturers list")

        response = self._make_request('GET', '/manufacturers')

        manufacturers = response.get('data', [])

        # Extract manufacturer names if response contains objects
        if manufacturers and isinstance(manufacturers[0], dict):
            manufacturers = [m.get('name') for m in manufacturers if m.get('name')]

        manufacturers = sorted(set(manufacturers))
        logger.info(f"Retrieved {len(manufacturers)} manufacturers")

        return manufacturers

    def filter_by_specifications(self, specs: Dict[str, Any]) -> List[Dict]:
        """
        Filter components by detailed technical specifications

        Args:
            specs: Dictionary of technical specifications to filter by:
                - category: str (required) - Component category
                - voltage_range: tuple (min_v, max_v)
                - current_range: tuple (min_a, max_a)
                - temperature_coefficient: float (max value)
                - dimensions: dict with 'max_length', 'max_width', 'max_height' in mm
                - weight_max: float (kg)
                - warranty_min: int (years)
                - certification: str or list (e.g., 'IEC61215', ['UL1703', 'CE'])
                - Any other technical parameters

        Returns:
            List of components matching all specified criteria

        Raises:
            ValueError: If required specifications are missing
            RateLimitError: If rate limit is exceeded
            APIError: If API request fails

        Example:
            >>> specs = {
            ...     'category': 'modules',
            ...     'voltage_range': (30, 50),
            ...     'warranty_min': 25,
            ...     'certification': ['IEC61215', 'UL1703']
            ... }
            >>> components = client.filter_by_specifications(specs)
        """
        if not specs or 'category' not in specs:
            raise ValueError("Specifications must include 'category'")

        category = specs['category']
        self._validate_category(category)

        logger.info(f"Filtering {category} by specifications: {specs}")

        # Build request payload
        payload = {
            'category': category,
            'specifications': {k: v for k, v in specs.items() if k != 'category'}
        }

        response = self._make_request('POST', '/components/filter', json=payload)

        results = response.get('data', [])
        logger.info(f"Found {len(results)} components matching specifications")

        return results

    def clear_cache(self):
        """
        Clear all cached data (e.g., manufacturer list cache)

        Useful when you need fresh data or after cache becomes stale.
        """
        self.list_manufacturers.cache_clear()
        logger.info("Cache cleared")

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get current rate limit status

        Returns:
            Dictionary with:
                - max_requests: Maximum requests allowed per hour
                - requests_made: Number of requests in current window
                - requests_remaining: Number of requests remaining
                - window_reset_seconds: Seconds until window resets
        """
        now = datetime.now()

        # Clean old requests
        while self.rate_limiter.requests and \
              (now - self.rate_limiter.requests[0]) > timedelta(seconds=self.rate_limiter.time_window):
            self.rate_limiter.requests.popleft()

        requests_made = len(self.rate_limiter.requests)
        requests_remaining = max(0, self.rate_limiter.max_requests - requests_made)

        window_reset_seconds = 0
        if self.rate_limiter.requests:
            oldest_request = self.rate_limiter.requests[0]
            window_reset = oldest_request + timedelta(seconds=self.rate_limiter.time_window)
            window_reset_seconds = max(0, (window_reset - now).total_seconds())

        return {
            'max_requests': self.rate_limiter.max_requests,
            'requests_made': requests_made,
            'requests_remaining': requests_remaining,
            'window_reset_seconds': int(window_reset_seconds)
        }

    def __repr__(self) -> str:
        """String representation of the client"""
        status = self.get_rate_limit_status()
        return (
            f"ENFSolarClient(base_url='{self.base_url}', "
            f"requests_remaining={status['requests_remaining']}/{status['max_requests']})"
        )

    def close(self):
        """Close the HTTP session"""
        if self.session:
            self.session.close()
            logger.info("ENFSolarClient session closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
