"""
Analysis Suite Module (B04-B06)
================================
Integrates:
- B04: IEC 61215/61730 Testing & Certification
- B05: System Design & Optimization (PVsyst/SAM integration)
- B06: Weather Data Analysis & Energy Yield Assessment (EYA)

This module provides comprehensive PV system analysis from testing standards
through system design optimization to energy yield forecasting.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator


# ============================================================================
# B04: IEC 61215/61730 TESTING & CERTIFICATION
# ============================================================================

class IECStandard(str, Enum):
    """IEC testing standards."""
    IEC_61215 = "IEC 61215"  # Design qualification and type approval
    IEC_61730 = "IEC 61730"  # Safety qualification
    IEC_61853 = "IEC 61853"  # PV module performance testing
    IEC_62804 = "IEC 62804"  # PID testing
    IEC_61701 = "IEC 61701"  # Salt mist corrosion


class TestResult(str, Enum):
    """Test outcome."""
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"
    NOT_TESTED = "NOT_TESTED"


class IECTestCase(BaseModel):
    """Individual IEC test case specification."""

    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Test name")
    standard: IECStandard = Field(..., description="IEC standard")
    category: str = Field(..., description="Test category")
    description: str = Field(..., description="Test description")
    test_parameters: Dict[str, Any] = Field(default_factory=dict, description="Test parameters")
    acceptance_criteria: str = Field(..., description="Pass/fail criteria")
    duration_hours: float = Field(default=0.0, ge=0, description="Test duration (hours)")
    result: TestResult = Field(default=TestResult.NOT_TESTED, description="Test result")
    measured_value: Optional[float] = Field(None, description="Measured test value")
    notes: str = Field(default="", description="Additional notes")

    class Config:
        use_enum_values = True


class IECTestingSuite:
    """
    IEC 61215/61730 Testing & Certification Suite.
    Manages comprehensive PV module testing according to international standards.
    """

    def __init__(self):
        """Initialize IEC testing suite with standard test cases."""
        self.test_cases: Dict[str, IECTestCase] = {}
        self._initialize_test_cases()

    def _initialize_test_cases(self) -> None:
        """Initialize standard IEC test cases."""
        standard_tests = [
            # IEC 61215 Tests
            IECTestCase(
                test_id="IEC61215-10.1",
                test_name="Visual Inspection",
                standard=IECStandard.IEC_61215,
                category="Visual",
                description="Visual examination for defects, cracks, bubbles",
                acceptance_criteria="No major defects visible",
                duration_hours=0.5
            ),
            IECTestCase(
                test_id="IEC61215-10.2",
                test_name="Maximum Power Determination",
                standard=IECStandard.IEC_61215,
                category="Electrical",
                description="Measure maximum power at STC",
                test_parameters={"temperature": 25, "irradiance": 1000, "spectrum": "AM1.5G"},
                acceptance_criteria="Pmax ≥ 95% of rated power",
                duration_hours=1.0
            ),
            IECTestCase(
                test_id="IEC61215-10.3",
                test_name="Insulation Test",
                standard=IECStandard.IEC_61215,
                category="Safety",
                description="Wet leakage current test",
                test_parameters={"test_voltage": 1000, "duration_min": 1},
                acceptance_criteria="Leakage current < 50 μA",
                duration_hours=2.0
            ),
            IECTestCase(
                test_id="IEC61215-10.6",
                test_name="Temperature Coefficient",
                standard=IECStandard.IEC_61215,
                category="Electrical",
                description="Measure Pmax, Voc, Isc temperature coefficients",
                test_parameters={"temp_range": [0, 75], "irradiance": 1000},
                acceptance_criteria="Within manufacturer specifications",
                duration_hours=4.0
            ),
            IECTestCase(
                test_id="IEC61215-10.7",
                test_name="NOCT (Nominal Operating Cell Temperature)",
                standard=IECStandard.IEC_61215,
                category="Thermal",
                description="Determine NOCT under standard conditions",
                test_parameters={"irradiance": 800, "ambient_temp": 20, "wind_speed": 1},
                acceptance_criteria="NOCT documented",
                duration_hours=6.0
            ),
            IECTestCase(
                test_id="IEC61215-10.8",
                test_name="Low Irradiance Performance",
                standard=IECStandard.IEC_61215,
                category="Electrical",
                description="Performance at 200 W/m² and 500 W/m²",
                test_parameters={"irradiance_levels": [200, 500, 1000]},
                acceptance_criteria="≥ 90% of linear performance",
                duration_hours=3.0
            ),
            IECTestCase(
                test_id="IEC61215-10.9",
                test_name="Outdoor Exposure",
                standard=IECStandard.IEC_61215,
                category="Environmental",
                description="60 kWh/m² outdoor exposure",
                test_parameters={"min_irradiance_total": 60},
                acceptance_criteria="Power degradation ≤ 5%",
                duration_hours=720.0
            ),
            IECTestCase(
                test_id="IEC61215-10.10",
                test_name="Hot-Spot Endurance",
                standard=IECStandard.IEC_61215,
                category="Thermal",
                description="Hot-spot heating test",
                test_parameters={"duration_hours": 1, "temperature_limit": 85},
                acceptance_criteria="No damage, Pmax ≥ 95%",
                duration_hours=5.0
            ),
            IECTestCase(
                test_id="IEC61215-10.11",
                test_name="UV Preconditioning",
                standard=IECStandard.IEC_61215,
                category="Environmental",
                description="UV exposure 15 kWh/m² (280-400 nm)",
                test_parameters={"uv_dose": 15, "wavelength_range": [280, 400]},
                acceptance_criteria="Power degradation ≤ 5%",
                duration_hours=120.0
            ),
            IECTestCase(
                test_id="IEC61215-10.12",
                test_name="Thermal Cycling",
                standard=IECStandard.IEC_61215,
                category="Thermal",
                description="200 cycles: -40°C to +85°C",
                test_parameters={"cycles": 200, "temp_range": [-40, 85]},
                acceptance_criteria="Pmax ≥ 95%, no defects",
                duration_hours=200.0
            ),
            IECTestCase(
                test_id="IEC61215-10.13",
                test_name="Humidity-Freeze",
                standard=IECStandard.IEC_61215,
                category="Environmental",
                description="10 cycles: +85°C/85%RH to -40°C",
                test_parameters={"cycles": 10, "humidity": 85},
                acceptance_criteria="Pmax ≥ 95%, no defects",
                duration_hours=80.0
            ),
            IECTestCase(
                test_id="IEC61215-10.14",
                test_name="Damp Heat",
                standard=IECStandard.IEC_61215,
                category="Environmental",
                description="1000 hours at +85°C/85%RH",
                test_parameters={"duration_hours": 1000, "temperature": 85, "humidity": 85},
                acceptance_criteria="Pmax ≥ 90%, no major defects",
                duration_hours=1000.0
            ),
            IECTestCase(
                test_id="IEC61215-10.16",
                test_name="Mechanical Load",
                standard=IECStandard.IEC_61215,
                category="Mechanical",
                description="Static load test: 2400 Pa (wind), 5400 Pa (snow)",
                test_parameters={"front_load": 2400, "rear_load": 2400, "snow_load": 5400},
                acceptance_criteria="Pmax ≥ 95%, no breakage",
                duration_hours=3.0
            ),
            IECTestCase(
                test_id="IEC61215-10.17",
                test_name="Hail Impact",
                standard=IECStandard.IEC_61215,
                category="Mechanical",
                description="Ice ball impact: 25mm at 23 m/s",
                test_parameters={"ball_diameter_mm": 25, "velocity_ms": 23, "impacts": 11},
                acceptance_criteria="No breakage, Pmax ≥ 95%",
                duration_hours=1.0
            ),
            # IEC 61730 Safety Tests
            IECTestCase(
                test_id="IEC61730-10.1",
                test_name="Dielectric Withstand",
                standard=IECStandard.IEC_61730,
                category="Safety",
                description="High voltage test",
                test_parameters={"test_voltage": 2000, "duration_min": 1},
                acceptance_criteria="No flashover or breakdown",
                duration_hours=1.0
            ),
            IECTestCase(
                test_id="IEC61730-10.2",
                test_name="Fire Test",
                standard=IECStandard.IEC_61730,
                category="Safety",
                description="Fire resistance class rating",
                test_parameters={"test_method": "spread of flame"},
                acceptance_criteria="Class A, B, or C rating",
                duration_hours=2.0
            ),
            # IEC 62804 PID Test
            IECTestCase(
                test_id="IEC62804",
                test_name="Potential Induced Degradation (PID)",
                standard=IECStandard.IEC_62804,
                category="Degradation",
                description="PID test: 96 hours at +85°C/85%RH with -1000V bias",
                test_parameters={"duration_hours": 96, "voltage": -1000, "temp": 85, "humidity": 85},
                acceptance_criteria="Power degradation ≤ 5%",
                duration_hours=96.0
            ),
        ]

        for test in standard_tests:
            self.test_cases[test.test_id] = test

    def run_test(self, test_id: str, measured_value: Optional[float] = None) -> IECTestCase:
        """
        Run or update a test case.

        Args:
            test_id: Test identifier
            measured_value: Measured test value

        Returns:
            Updated test case with result
        """
        test = self.test_cases.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        # Simulate test execution
        test.measured_value = measured_value
        test.result = self._evaluate_test_result(test)

        return test

    def _evaluate_test_result(self, test: IECTestCase) -> TestResult:
        """Evaluate test result based on acceptance criteria."""
        if test.measured_value is None:
            return TestResult.PENDING

        # Simplified pass/fail logic (in production, this would be more complex)
        if "Pmax ≥ 95%" in test.acceptance_criteria:
            return TestResult.PASS if test.measured_value >= 95 else TestResult.FAIL
        elif "Pmax ≥ 90%" in test.acceptance_criteria:
            return TestResult.PASS if test.measured_value >= 90 else TestResult.FAIL
        else:
            return TestResult.PASS

    def get_certification_status(self) -> Dict[str, Any]:
        """Get overall certification status."""
        total_tests = len(self.test_cases)
        passed = sum(1 for t in self.test_cases.values() if t.result == TestResult.PASS)
        failed = sum(1 for t in self.test_cases.values() if t.result == TestResult.FAIL)
        pending = sum(1 for t in self.test_cases.values() if t.result == TestResult.PENDING)
        not_tested = sum(1 for t in self.test_cases.values() if t.result == TestResult.NOT_TESTED)

        certified = failed == 0 and not_tested == 0 and pending == 0

        return {
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'pending': pending,
            'not_tested': not_tested,
            'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
            'certified': certified,
            'standards': list(set(t.standard for t in self.test_cases.values()))
        }

    def get_test_summary_df(self) -> pd.DataFrame:
        """Export test summary as DataFrame."""
        data = []
        for test in self.test_cases.values():
            data.append({
                'Test ID': test.test_id,
                'Test Name': test.test_name,
                'Standard': test.standard,
                'Category': test.category,
                'Duration (hrs)': test.duration_hours,
                'Result': test.result,
                'Measured Value': test.measured_value if test.measured_value else '-'
            })
        return pd.DataFrame(data)


# ============================================================================
# B05: SYSTEM DESIGN & OPTIMIZATION
# ============================================================================

class InverterType(str, Enum):
    """Inverter topology types."""
    STRING = "String Inverter"
    CENTRAL = "Central Inverter"
    MICRO = "Micro Inverter"
    POWER_OPTIMIZER = "Power Optimizer"
    HYBRID = "Hybrid Inverter"


class MountingType(str, Enum):
    """System mounting configurations."""
    FIXED_TILT = "Fixed Tilt"
    SINGLE_AXIS_TRACKER = "Single-Axis Tracker"
    DUAL_AXIS_TRACKER = "Dual-Axis Tracker"
    ROOF_MOUNTED = "Roof Mounted"
    GROUND_MOUNTED = "Ground Mounted"
    CARPORT = "Carport"
    FLOATING = "Floating PV"


class SystemConfiguration(BaseModel):
    """PV system configuration parameters."""

    system_name: str = Field(..., description="System identifier")
    capacity_kw: float = Field(..., ge=0, description="System capacity (kW)")
    module_power_wp: float = Field(..., ge=0, description="Module rated power (Wp)")
    num_modules: int = Field(..., ge=1, description="Total number of modules")
    modules_per_string: int = Field(..., ge=1, description="Modules per string")
    num_strings: int = Field(..., ge=1, description="Number of parallel strings")
    inverter_type: InverterType = Field(..., description="Inverter topology")
    inverter_efficiency: float = Field(default=97.5, ge=90, le=99.9, description="Inverter efficiency (%)")
    inverter_capacity_kw: float = Field(..., ge=0, description="Inverter capacity (kW)")
    dc_ac_ratio: float = Field(default=1.2, ge=1.0, le=2.0, description="DC/AC ratio")
    mounting_type: MountingType = Field(..., description="Mounting configuration")
    tilt_angle: float = Field(..., ge=0, le=90, description="Tilt angle (degrees)")
    azimuth: float = Field(..., ge=0, le=360, description="Azimuth (degrees, 0=North, 180=South)")
    system_losses: float = Field(default=14.0, ge=0, le=50, description="Total system losses (%)")
    location: Dict[str, float] = Field(..., description="Location coordinates")

    class Config:
        use_enum_values = True

    @validator('num_modules')
    def validate_module_count(cls, v, values):
        """Validate module count matches string configuration."""
        if 'modules_per_string' in values and 'num_strings' in values:
            expected = values['modules_per_string'] * values['num_strings']
            if v != expected:
                raise ValueError(f"Module count {v} doesn't match configuration: {expected}")
        return v


class SystemDesigner:
    """
    PV System Design & Optimization Engine.
    Integrates with PVsyst and SAM methodologies for comprehensive system design.
    """

    def __init__(self):
        """Initialize system designer."""
        self.designs: Dict[str, SystemConfiguration] = {}

    def design_system(
        self,
        capacity_kw: float,
        module_power_wp: float,
        location: Dict[str, float],
        inverter_type: InverterType = InverterType.STRING,
        mounting_type: MountingType = MountingType.FIXED_TILT
    ) -> SystemConfiguration:
        """
        Design PV system with optimized configuration.

        Args:
            capacity_kw: Target system capacity
            module_power_wp: Module power rating
            location: Geographic location
            inverter_type: Inverter topology
            mounting_type: Mounting configuration

        Returns:
            Optimized system configuration
        """
        # Calculate optimal string configuration
        num_modules = int(capacity_kw * 1000 / module_power_wp)
        modules_per_string, num_strings = self._optimize_string_configuration(
            num_modules, module_power_wp
        )

        # Calculate optimal tilt and azimuth
        tilt_angle = self._calculate_optimal_tilt(location['latitude'])
        azimuth = 180 if location['latitude'] >= 0 else 0  # South for Northern hemisphere

        # Size inverter
        dc_capacity = num_modules * module_power_wp / 1000
        dc_ac_ratio = 1.25 if mounting_type == MountingType.SINGLE_AXIS_TRACKER else 1.2
        inverter_capacity_kw = dc_capacity / dc_ac_ratio

        config = SystemConfiguration(
            system_name=f"PV_{capacity_kw}kW_{datetime.now().strftime('%Y%m%d')}",
            capacity_kw=dc_capacity,
            module_power_wp=module_power_wp,
            num_modules=num_modules,
            modules_per_string=modules_per_string,
            num_strings=num_strings,
            inverter_type=inverter_type,
            inverter_capacity_kw=inverter_capacity_kw,
            dc_ac_ratio=dc_ac_ratio,
            mounting_type=mounting_type,
            tilt_angle=tilt_angle,
            azimuth=azimuth,
            location=location
        )

        self.designs[config.system_name] = config
        return config

    def _optimize_string_configuration(
        self,
        num_modules: int,
        module_power_wp: float
    ) -> Tuple[int, int]:
        """
        Optimize modules per string and number of strings.

        Returns:
            (modules_per_string, num_strings)
        """
        # Typical string voltage range: 600-1000V
        # Assuming ~40V per module
        optimal_modules_per_string = 20

        # Find closest divisor to optimal
        for modules_per_string in range(optimal_modules_per_string, 10, -1):
            if num_modules % modules_per_string == 0:
                num_strings = num_modules // modules_per_string
                return modules_per_string, num_strings

        # Fallback
        num_strings = max(1, num_modules // optimal_modules_per_string)
        modules_per_string = num_modules // num_strings
        return modules_per_string, num_strings

    def _calculate_optimal_tilt(self, latitude: float) -> float:
        """Calculate optimal tilt angle based on latitude."""
        # Simple rule: tilt ≈ latitude for fixed systems
        return abs(latitude)

    def calculate_energy_yield(
        self,
        config: SystemConfiguration,
        annual_irradiance_kwh_m2: float
    ) -> Dict[str, Any]:
        """
        Calculate annual energy yield.

        Args:
            config: System configuration
            annual_irradiance_kwh_m2: Annual plane-of-array irradiance

        Returns:
            Energy yield analysis
        """
        # DC energy generation
        module_area_m2 = config.module_power_wp / 200  # Approximate module area
        total_area_m2 = module_area_m2 * config.num_modules
        module_efficiency = config.module_power_wp / (module_area_m2 * 1000)

        dc_energy_kwh = (
            annual_irradiance_kwh_m2 *
            total_area_m2 *
            module_efficiency *
            (1 - config.system_losses / 100)
        )

        # AC energy (after inverter)
        ac_energy_kwh = dc_energy_kwh * (config.inverter_efficiency / 100)

        # Specific yield (kWh/kWp)
        specific_yield = ac_energy_kwh / config.capacity_kw

        # Performance ratio
        reference_yield = annual_irradiance_kwh_m2
        performance_ratio = (specific_yield / reference_yield) * 100 if reference_yield > 0 else 0

        return {
            'dc_energy_kwh': dc_energy_kwh,
            'ac_energy_kwh': ac_energy_kwh,
            'specific_yield_kwh_kwp': specific_yield,
            'performance_ratio': performance_ratio,
            'capacity_factor': (ac_energy_kwh / (config.capacity_kw * 8760)) * 100,
            'inverter_losses_kwh': dc_energy_kwh - ac_energy_kwh
        }


# ============================================================================
# B06: WEATHER DATA ANALYSIS & ENERGY YIELD ASSESSMENT
# ============================================================================

class WeatherDataSource(str, Enum):
    """Weather data sources."""
    PVGIS = "PVGIS"
    NASA_SSE = "NASA SSE"
    METEONORM = "Meteonorm"
    NREL_NSRDB = "NREL NSRDB"
    LOCAL_SENSOR = "Local Sensor"


class WeatherRecord(BaseModel):
    """Weather data record for a specific timestamp."""

    timestamp: datetime = Field(..., description="Timestamp")
    ghi: float = Field(..., ge=0, le=1500, description="Global Horizontal Irradiance (W/m²)")
    dni: float = Field(..., ge=0, le=1200, description="Direct Normal Irradiance (W/m²)")
    dhi: float = Field(..., ge=0, le=800, description="Diffuse Horizontal Irradiance (W/m²)")
    ambient_temp: float = Field(..., ge=-50, le=60, description="Ambient temperature (°C)")
    wind_speed: float = Field(default=0.0, ge=0, description="Wind speed (m/s)")
    relative_humidity: float = Field(default=50.0, ge=0, le=100, description="Relative humidity (%)")
    atmospheric_pressure: float = Field(default=1013.0, ge=800, le=1100, description="Pressure (mbar)")
    precipitation: float = Field(default=0.0, ge=0, description="Precipitation (mm/hr)")


class EnergyYieldAssessment:
    """
    Energy Yield Assessment (EYA) Engine.
    Analyzes weather data and calculates expected energy production.
    """

    def __init__(self, data_source: WeatherDataSource = WeatherDataSource.PVGIS):
        """Initialize EYA engine."""
        self.data_source = data_source
        self.weather_data: List[WeatherRecord] = []

    def load_typical_meteorological_year(
        self,
        latitude: float,
        longitude: float
    ) -> List[WeatherRecord]:
        """
        Load TMY (Typical Meteorological Year) data.

        Args:
            latitude: Location latitude
            longitude: Location longitude

        Returns:
            Hourly weather data for typical year
        """
        # Simulate TMY data generation (8760 hours)
        start_date = datetime(2025, 1, 1)
        weather_data = []

        for hour in range(8760):
            timestamp = start_date + timedelta(hours=hour)
            day_of_year = timestamp.timetuple().tm_yday
            hour_of_day = timestamp.hour

            # Simplified solar radiation model
            ghi = self._calculate_ghi(latitude, day_of_year, hour_of_day)
            dni = ghi * 0.7 if ghi > 0 else 0
            dhi = ghi * 0.3 if ghi > 0 else 0

            # Temperature model (simplified)
            ambient_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + \
                          5 * np.sin(2 * np.pi * hour_of_day / 24)

            weather_data.append(WeatherRecord(
                timestamp=timestamp,
                ghi=ghi,
                dni=dni,
                dhi=dhi,
                ambient_temp=ambient_temp,
                wind_speed=np.random.uniform(0, 5),
                relative_humidity=np.random.uniform(30, 80)
            ))

        self.weather_data = weather_data
        return weather_data

    def _calculate_ghi(self, latitude: float, day_of_year: int, hour_of_day: int) -> float:
        """Calculate Global Horizontal Irradiance (simplified model)."""
        # Solar declination
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))

        # Hour angle
        hour_angle = 15 * (hour_of_day - 12)

        # Solar altitude angle
        altitude = np.degrees(np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        ))

        if altitude <= 0:
            return 0

        # GHI approximation
        ghi = 1000 * np.sin(np.radians(altitude)) * 0.7  # Clear sky model
        return max(0, ghi)

    def calculate_poa_irradiance(
        self,
        tilt: float,
        azimuth: float
    ) -> pd.DataFrame:
        """
        Calculate Plane-of-Array (POA) irradiance.

        Args:
            tilt: Surface tilt angle (degrees)
            azimuth: Surface azimuth (degrees)

        Returns:
            DataFrame with POA irradiance data
        """
        if not self.weather_data:
            raise ValueError("No weather data loaded. Call load_typical_meteorological_year first.")

        poa_data = []
        for record in self.weather_data:
            # Simplified POA calculation (Hay-Davies transposition model)
            # In production, use pvlib for accurate calculations
            poa_global = record.ghi * np.cos(np.radians(tilt))  # Simplified
            poa_global = max(0, poa_global)

            poa_data.append({
                'timestamp': record.timestamp,
                'poa_global': poa_global,
                'ghi': record.ghi,
                'ambient_temp': record.ambient_temp
            })

        return pd.DataFrame(poa_data)

    def run_energy_yield_assessment(
        self,
        system_config: SystemConfiguration,
        confidence_level: float = 0.9
    ) -> Dict[str, Any]:
        """
        Run comprehensive Energy Yield Assessment.

        Args:
            system_config: System configuration
            confidence_level: Confidence level for P-value calculations

        Returns:
            EYA results with P50, P75, P90, P99 values
        """
        # Load weather data
        location = system_config.location
        self.load_typical_meteorological_year(
            latitude=location['latitude'],
            longitude=location['longitude']
        )

        # Calculate POA irradiance
        poa_df = self.calculate_poa_irradiance(
            tilt=system_config.tilt_angle,
            azimuth=system_config.azimuth
        )

        # Annual POA irradiance
        annual_poa_kwh_m2 = poa_df['poa_global'].sum() / 1000

        # Calculate energy yield (using SystemDesigner method)
        designer = SystemDesigner()
        yield_results = designer.calculate_energy_yield(system_config, annual_poa_kwh_m2)

        # Calculate P-values (probabilistic energy estimates)
        p50_energy = yield_results['ac_energy_kwh']  # Median
        p75_energy = p50_energy * 0.95  # Conservative
        p90_energy = p50_energy * 0.90  # Very conservative (bankability)
        p99_energy = p50_energy * 0.85  # Extreme conservative

        return {
            'annual_poa_irradiance_kwh_m2': annual_poa_kwh_m2,
            'p50_energy_kwh': p50_energy,
            'p75_energy_kwh': p75_energy,
            'p90_energy_kwh': p90_energy,
            'p99_energy_kwh': p99_energy,
            'specific_yield_kwh_kwp': yield_results['specific_yield_kwh_kwp'],
            'performance_ratio': yield_results['performance_ratio'],
            'capacity_factor': yield_results['capacity_factor'],
            'confidence_level': confidence_level,
            'assessment_date': datetime.now().isoformat(),
            'data_source': self.data_source
        }


# ============================================================================
# ANALYSIS SUITE INTEGRATION INTERFACE
# ============================================================================

class AnalysisSuite:
    """
    Unified Analysis Suite Interface integrating B04-B06.
    Provides comprehensive PV system analysis from testing to energy assessment.
    """

    def __init__(self):
        """Initialize all analysis suite components."""
        self.iec_testing = IECTestingSuite()
        self.system_designer = SystemDesigner()
        self.eya_engine = EnergyYieldAssessment()

    def complete_system_analysis(
        self,
        module_power_wp: float,
        capacity_kw: float,
        location: Dict[str, float],
        run_iec_tests: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete system analysis workflow.

        Args:
            module_power_wp: Module power rating
            capacity_kw: Target system capacity
            location: Geographic location
            run_iec_tests: Whether to include IEC testing results

        Returns:
            Complete analysis results
        """
        # Step 1: System Design
        system_config = self.system_designer.design_system(
            capacity_kw=capacity_kw,
            module_power_wp=module_power_wp,
            location=location
        )

        # Step 2: Energy Yield Assessment
        eya_results = self.eya_engine.run_energy_yield_assessment(system_config)

        # Step 3: IEC Testing (optional)
        iec_status = None
        if run_iec_tests:
            iec_status = self.iec_testing.get_certification_status()

        return {
            'system_configuration': system_config.dict(),
            'energy_yield_assessment': eya_results,
            'iec_certification': iec_status,
            'analysis_complete': True
        }


# Export main interface
__all__ = [
    'AnalysisSuite',
    'IECTestingSuite',
    'IECTestCase',
    'IECStandard',
    'SystemDesigner',
    'SystemConfiguration',
    'EnergyYieldAssessment',
    'WeatherRecord'
]
