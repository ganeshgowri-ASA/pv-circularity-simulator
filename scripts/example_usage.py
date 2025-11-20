#!/usr/bin/env python3
"""
Example usage of PV Simulator TMY Weather Database.

This script demonstrates how to use the TMY data management system
to fetch and analyze weather data.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pv_simulator.api.nsrdb_client import NSRDBClient
from pv_simulator.api.pvgis_client import PVGISClient
from pv_simulator.config.settings import settings
from pv_simulator.services.global_coverage import GlobalWeatherCoverage
from pv_simulator.services.tmy_manager import TMYDataManager
from pv_simulator.services.weather_database import WeatherDatabaseBuilder


def example_nsrdb_usage() -> None:
    """Example: Fetch TMY data from NREL NSRDB."""
    print("=" * 60)
    print("Example 1: Fetching TMY data from NREL NSRDB")
    print("=" * 60)

    # Initialize weather database builder
    weather_db = WeatherDatabaseBuilder()

    # Fetch TMY data for Denver, CO
    print("\nFetching TMY data for Denver, CO...")
    try:
        tmy_data = weather_db.nrel_nsrdb_integration(
            latitude=39.7392, longitude=-104.9903, year=None  # None = TMY data
        )

        print(f"\n✓ Successfully fetched TMY data!")
        print(f"  Location: {tmy_data.location.name}")
        print(f"  Data source: {tmy_data.data_source.value}")
        print(f"  Data points: {len(tmy_data.hourly_data)}")
        print(f"  Quality: {tmy_data.data_quality.value}")
        print(f"  Completeness: {tmy_data.completeness_percentage:.1f}%")
        print(f"  Annual GHI: {tmy_data.get_annual_irradiation():.1f} kWh/m²")
        print(f"  Average Temperature: {tmy_data.get_average_temperature():.1f}°C")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Note: You may need to set NSRDB_API_KEY in .env file")


def example_pvgis_usage() -> None:
    """Example: Fetch TMY data from PVGIS."""
    print("\n" + "=" * 60)
    print("Example 2: Fetching TMY data from PVGIS")
    print("=" * 60)

    weather_db = WeatherDatabaseBuilder()

    # Fetch TMY data for Berlin, Germany
    print("\nFetching TMY data for Berlin, Germany...")
    try:
        tmy_data = weather_db.pvgis_data_fetcher(latitude=52.52, longitude=13.40)

        print(f"\n✓ Successfully fetched TMY data!")
        print(f"  Location: {tmy_data.location.name}")
        print(f"  Data source: {tmy_data.data_source.value}")
        print(f"  Data points: {len(tmy_data.hourly_data)}")
        print(f"  Annual GHI: {tmy_data.get_annual_irradiation():.1f} kWh/m²")

    except Exception as e:
        print(f"\n✗ Error: {e}")


def example_location_search() -> None:
    """Example: Search for locations in database."""
    print("\n" + "=" * 60)
    print("Example 3: Searching location database")
    print("=" * 60)

    coverage = GlobalWeatherCoverage()

    # Search by name
    print("\nSearching for 'Denver'...")
    results = coverage.search_by_name("Denver")

    if results:
        print(f"\n✓ Found {len(results)} locations:")
        for loc in results:
            print(
                f"  - {loc.name}, {loc.country} "
                f"({loc.latitude:.2f}, {loc.longitude:.2f})"
            )
    else:
        print("  No locations found")

    # Find nearest station
    print("\nFinding nearest station to coordinates (40.0, -105.0)...")
    nearest = coverage.nearest_station_finder(latitude=40.0, longitude=-105.0, num_stations=3)

    if nearest:
        print(f"\n✓ Found {len(nearest)} nearby stations:")
        for loc, distance in nearest:
            print(
                f"  - {loc.name}, {loc.country} "
                f"({loc.latitude:.2f}, {loc.longitude:.2f}) "
                f"- {distance:.1f} km away"
            )


def example_tmy_file_parsing() -> None:
    """Example: Parse TMY file from disk."""
    print("\n" + "=" * 60)
    print("Example 4: Parsing TMY files")
    print("=" * 60)

    tmy_manager = TMYDataManager()

    print("\nTMY file parsing example:")
    print("  Supported formats: TMY2, TMY3, EPW, CSV, JSON")
    print("  Use: tmy_data = tmy_manager.load_tmy_data('path/to/file.csv')")
    print("\nFor actual file parsing, provide a valid TMY file path.")


def main() -> None:
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print("PV Simulator - TMY Weather Database Examples")
    print("*" * 60)

    print(f"\nConfiguration:")
    print(f"  NSRDB API Key: {settings.nsrdb_api_key[:10]}...")
    print(f"  TMY Cache Dir: {settings.tmy_cache_dir}")
    print(f"  Weather Data Dir: {settings.weather_data_dir}")

    # Run examples
    example_nsrdb_usage()
    example_pvgis_usage()
    example_location_search()
    example_tmy_file_parsing()

    print("\n" + "*" * 60)
    print("Examples completed!")
    print("*" * 60 + "\n")


if __name__ == "__main__":
    main()
