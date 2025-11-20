"""
Example usage of AssetManager for PV Circularity Simulator.

This example demonstrates:
1. Database initialization
2. Creating sites and equipment
3. Tracking performance data
4. Using inventory and tracking methods
5. Circular economy features
"""

from datetime import datetime, timedelta
from src.pv_circularity.database.session import init_db
from src.pv_circularity.managers.asset_manager import AssetManager
from src.pv_circularity.database.models import AssetStatus, EquipmentType
from src.pv_circularity.models.schemas import (
    SiteCreate,
    EquipmentCreate,
    PerformanceRecordCreate,
)


def main():
    """Main example function."""
    print("=" * 80)
    print("PV Circularity Simulator - Asset Management Example")
    print("=" * 80)

    # 1. Initialize database
    print("\n1. Initializing database...")
    db_manager = init_db("sqlite:///./example_pv_circularity.db", create_tables=True)
    db_session = next(db_manager.get_session())
    asset_manager = AssetManager(db_session)
    print("   Database initialized successfully!")

    # 2. Create a solar installation site
    print("\n2. Creating solar installation site...")
    site_data = SiteCreate(
        name="Green Energy Solar Farm",
        location="Phoenix, Arizona, USA",
        latitude=33.4484,
        longitude=-112.0740,
        capacity_kw=5000.0,
        installation_date=datetime(2020, 3, 15),
        status=AssetStatus.ACTIVE,
        description="Large-scale solar farm in Arizona desert",
        metadata={
            "climate_zone": "hot-dry",
            "grid_connection": "active",
            "land_area_acres": 25,
        },
    )
    site = asset_manager.create_site(site_data)
    print(f"   Created site: {site.name} (ID: {site.id})")
    print(f"   Location: {site.location}")
    print(f"   Capacity: {site.capacity_kw} kW")

    # 3. Add solar panels to the site
    print("\n3. Adding solar panels...")
    panels = []
    for i in range(10):
        panel_data = EquipmentCreate(
            equipment_id=f"PANEL-{i+1:04d}",
            site_id=site.id,
            equipment_type=EquipmentType.SOLAR_PANEL,
            name=f"High Efficiency Panel {i+1}",
            manufacturer="SolarTech Industries",
            model="ST-550W-PERC",
            serial_number=f"ST550-2020-{1000+i}",
            status=AssetStatus.ACTIVE,
            rated_power_w=550.0,
            efficiency_percent=22.8,
            degradation_rate_percent=0.45,
            temperature_coefficient=-0.35,
            manufacturing_date=datetime(2020, 1, 15),
            installation_date=datetime(2020, 3, 15),
            warranty_expiry=datetime(2030, 3, 15),
            expected_lifetime_years=25.0,
            purchase_cost=275.0,
            current_value=250.0,
            recyclable=True,
            material_composition={
                "silicon": 0.35,
                "glass": 0.30,
                "aluminum": 0.20,
                "copper": 0.05,
                "other": 0.10,
            },
            recycling_value=40.0,
            description="PERC monocrystalline solar panel",
        )
        panel = asset_manager.create_equipment(panel_data)
        panels.append(panel)
    print(f"   Created {len(panels)} solar panels")

    # 4. Add inverters
    print("\n4. Adding inverters...")
    inverters = []
    for i in range(2):
        inverter_data = EquipmentCreate(
            equipment_id=f"INV-{i+1:04d}",
            site_id=site.id,
            equipment_type=EquipmentType.INVERTER,
            name=f"String Inverter {i+1}",
            manufacturer="PowerConvert Inc.",
            model="PC-50kW",
            serial_number=f"PC50-2020-{5000+i}",
            status=AssetStatus.ACTIVE,
            rated_power_w=50000.0,
            efficiency_percent=98.5,
            installation_date=datetime(2020, 3, 15),
            warranty_expiry=datetime(2030, 3, 15),
            expected_lifetime_years=15.0,
            purchase_cost=5000.0,
            current_value=4200.0,
            recyclable=True,
            material_composition={
                "electronics": 0.40,
                "metals": 0.35,
                "plastics": 0.15,
                "other": 0.10,
            },
            recycling_value=500.0,
        )
        inverter = asset_manager.create_equipment(inverter_data)
        inverters.append(inverter)
    print(f"   Created {len(inverters)} inverters")

    # 5. Record performance data
    print("\n5. Recording performance data...")
    base_date = datetime(2023, 6, 1, 12, 0, 0)
    for day in range(30):
        timestamp = base_date + timedelta(days=day)
        # Simulate varying daily production
        energy_kwh = 4500.0 + (day % 7) * 200 - (day % 3) * 100
        power_kw = energy_kwh / 8.0  # Assuming 8 hours of production

        perf_data = PerformanceRecordCreate(
            site_id=site.id,
            equipment_id=None,  # Site-level record
            timestamp=timestamp,
            energy_generated_kwh=energy_kwh,
            power_output_kw=power_kw,
            efficiency_percent=22.5 - (day % 10) * 0.1,
            capacity_factor_percent=80.0 + (day % 5) * 2,
            performance_ratio=0.92 + (day % 8) * 0.01,
            irradiance_w_m2=950.0 + (day % 6) * 20,
            temperature_c=32.0 + (day % 4) * 3,
            wind_speed_ms=2.5 + (day % 5) * 0.5,
            availability_percent=99.0 + (day % 10) * 0.1,
            downtime_hours=0.2 if day % 7 == 0 else 0.0,
        )
        asset_manager.create_performance_record(perf_data)
    print(f"   Recorded 30 days of performance data")

    # 6. Demonstrate site_inventory() method
    print("\n6. Site Inventory Summary:")
    print("-" * 80)
    inventory = asset_manager.site_inventory(include_summary=True)
    summary = inventory["summary"]
    print(f"   Total Sites: {summary.total_sites}")
    print(f"   Total Capacity: {summary.total_capacity_kw:.2f} kW")
    print(f"   Average Capacity: {summary.average_capacity_kw:.2f} kW")
    print(f"   Sites by Status: {summary.sites_by_status}")

    # 7. Demonstrate equipment_tracking() method
    print("\n7. Equipment Tracking Summary:")
    print("-" * 80)
    tracking = asset_manager.equipment_tracking(include_summary=True)
    eq_summary = tracking["summary"]
    print(f"   Total Equipment: {eq_summary.total_equipment}")
    print(f"   Total Rated Power: {eq_summary.total_rated_power_kw:.2f} kW")
    print(f"   Average Efficiency: {eq_summary.average_efficiency_percent:.2f}%")
    print(f"   Equipment by Type: {eq_summary.equipment_by_type}")
    print(f"   Equipment by Status: {eq_summary.equipment_by_status}")

    # 8. Demonstrate performance_history() method
    print("\n8. Performance History Summary:")
    print("-" * 80)
    history = asset_manager.performance_history(include_summary=True)
    perf_summary = history["summary"]
    print(f"   Total Records: {perf_summary.total_records}")
    print(f"   Date Range: {perf_summary.date_range['start'].date()} to {perf_summary.date_range['end'].date()}")
    print(f"   Total Energy Generated: {perf_summary.total_energy_kwh:.2f} kWh")
    print(f"   Average Power Output: {perf_summary.average_power_kw:.2f} kW")
    print(f"   Average Efficiency: {perf_summary.average_efficiency_percent:.2f}%")
    print(f"   Average Capacity Factor: {perf_summary.average_capacity_factor_percent:.2f}%")

    # 9. Filter performance by date range
    print("\n9. Performance History (Last 7 Days):")
    print("-" * 80)
    end_date = base_date + timedelta(days=29)
    start_date = end_date - timedelta(days=7)
    recent_history = asset_manager.performance_history(
        start_date=start_date, end_date=end_date, include_summary=True
    )
    recent_summary = recent_history["summary"]
    print(f"   Records: {recent_summary.total_records}")
    print(f"   Energy Generated: {recent_summary.total_energy_kwh:.2f} kWh")
    print(f"   Average Power: {recent_summary.average_power_kw:.2f} kW")

    # 10. Circular economy features
    print("\n10. Circular Economy Analysis:")
    print("-" * 80)
    all_equipment = asset_manager.list_equipment(site_id=site.id)
    total_recycling_value = sum(eq.recycling_value or 0 for eq in all_equipment)
    recyclable_count = sum(1 for eq in all_equipment if eq.recyclable)

    print(f"   Total Equipment: {len(all_equipment)}")
    print(f"   Recyclable Equipment: {recyclable_count} ({recyclable_count/len(all_equipment)*100:.1f}%)")
    print(f"   Total Recycling Value: ${total_recycling_value:.2f}")

    # Show material composition for first panel
    if panels:
        print(f"\n   Sample Panel Material Composition:")
        for material, percentage in panels[0].material_composition.items():
            print(f"     - {material.capitalize()}: {percentage*100:.1f}%")

    # 11. Lifecycle tracking example
    print("\n11. Equipment Lifecycle Example:")
    print("-" * 80)
    sample_panel = panels[0]
    install_date = sample_panel.installation_date
    expected_end = install_date.replace(year=install_date.year + int(sample_panel.expected_lifetime_years))
    years_in_operation = (datetime.now() - install_date).days / 365.25
    remaining_years = sample_panel.expected_lifetime_years - years_in_operation

    print(f"   Equipment: {sample_panel.name}")
    print(f"   Installation Date: {install_date.date()}")
    print(f"   Expected Lifetime: {sample_panel.expected_lifetime_years} years")
    print(f"   Expected End of Life: {expected_end.date()}")
    print(f"   Years in Operation: {years_in_operation:.1f}")
    print(f"   Remaining Years: {remaining_years:.1f}")
    print(f"   Current Value: ${sample_panel.current_value:.2f}")
    print(f"   Degradation Rate: {sample_panel.degradation_rate_percent}% per year")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)

    # Clean up
    db_session.close()


if __name__ == "__main__":
    main()
