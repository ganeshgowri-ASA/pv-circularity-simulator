"""Main system design engine orchestrating all PV system design components."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from pv_simulator.system_design.models import (
    ModuleParameters,
    InverterParameters,
    SystemConfiguration,
    SystemType,
    MountingType,
    ArrayLayout,
    StringConfiguration,
    SystemLosses,
)
from pv_simulator.system_design.string_sizing_calculator import StringSizingCalculator
from pv_simulator.system_design.inverter_selector import InverterSelector
from pv_simulator.system_design.array_layout_designer import ArrayLayoutDesigner
from pv_simulator.system_design.system_loss_model import SystemLossModel

logger = logging.getLogger(__name__)


class SystemDesignEngine:
    """
    Main PV system design engine.

    Orchestrates module selection, string sizing, inverter selection, array layout,
    and loss calculations to create complete PV system designs for utility-scale,
    commercial, and residential applications.
    """

    def __init__(
        self,
        project_name: str,
        system_type: SystemType = SystemType.UTILITY,
        location: str = "Unknown",
        latitude: float = 35.0,
        longitude: float = -100.0,
        elevation: float = 0.0,
    ):
        """
        Initialize system design engine.

        Args:
            project_name: Name of the project
            system_type: Type of system (utility/commercial/residential)
            location: Location name
            latitude: Site latitude (degrees)
            longitude: Site longitude (degrees)
            elevation: Site elevation (meters)
        """
        self.project_name = project_name
        self.system_type = system_type
        self.location = location
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

        # Component instances (initialized when module/inverter selected)
        self.string_calculator: Optional[StringSizingCalculator] = None
        self.inverter_selector: Optional[InverterSelector] = None
        self.layout_designer: Optional[ArrayLayoutDesigner] = None
        self.loss_model: Optional[SystemLossModel] = None

        logger.info(
            f"Initialized SystemDesignEngine: {project_name} "
            f"({system_type.value}) at {location}"
        )

    def design_system_configuration(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        target_dc_capacity_kw: float,
        mounting_type: MountingType = MountingType.GROUND_FIXED,
        site_temp_min: float = -10.0,
        site_temp_max: float = 70.0,
        target_dc_ac_ratio: float = 1.25,
        **kwargs,
    ) -> SystemConfiguration:
        """
        Design complete system configuration.

        Args:
            module: Module parameters
            inverter: Inverter parameters
            target_dc_capacity_kw: Target DC capacity in kW
            mounting_type: Type of mounting system
            site_temp_min: Minimum site temperature (°C)
            site_temp_max: Maximum site temperature (°C)
            target_dc_ac_ratio: Target DC/AC ratio
            **kwargs: Additional configuration parameters

        Returns:
            Complete SystemConfiguration
        """
        logger.info(f"Designing {target_dc_capacity_kw}kW system configuration")

        # Initialize components
        self.string_calculator = StringSizingCalculator(
            module=module,
            inverter=inverter,
            site_temp_min=site_temp_min,
            site_temp_max=site_temp_max,
        )

        self.inverter_selector = InverterSelector(
            module=module,
            system_type=self.system_type,
        )

        self.layout_designer = ArrayLayoutDesigner(
            module=module,
            mounting_type=mounting_type,
            site_latitude=self.latitude,
        )

        self.loss_model = SystemLossModel(
            module=module,
            inverter=inverter,
            mounting_type=mounting_type,
        )

        # Design string configuration
        string_config = self.calculate_string_sizing(
            module=module,
            inverter=inverter,
            site_temp_min=site_temp_min,
            site_temp_max=site_temp_max,
        )

        # Calculate number of modules and inverters
        num_modules = int((target_dc_capacity_kw * 1000) / module.pmax)

        # Calculate number of inverters based on DC/AC ratio
        target_ac_capacity_kw = target_dc_capacity_kw / target_dc_ac_ratio
        num_inverters = max(1, int(np.ceil((target_ac_capacity_kw * 1000) / inverter.pac_max)))

        # Design array layout
        array_layout = self.design_array_layout(
            total_modules=num_modules,
            mounting_type=mounting_type,
            modules_per_string=string_config.modules_per_string,
            **kwargs,
        )

        # Calculate system losses
        losses = self.calculate_system_losses(
            module=module,
            inverter=inverter,
            mounting_type=mounting_type,
            **kwargs,
        )

        # Create system configuration
        system_config = SystemConfiguration(
            project_name=self.project_name,
            system_type=self.system_type,
            location=self.location,
            latitude=self.latitude,
            longitude=self.longitude,
            elevation=self.elevation,
            design_date=datetime.now(),
            site_temp_min=site_temp_min,
            site_temp_max=site_temp_max,
            avg_ambient_temp=kwargs.get('avg_ambient_temp', (site_temp_min + site_temp_max) / 2),
            module=module,
            inverter=inverter,
            array_layout=array_layout,
            string_config=string_config,
            num_modules=num_modules,
            num_inverters=num_inverters,
            losses=losses,
        )

        # Calculate derived parameters
        system_config.dc_capacity = system_config.calculate_dc_capacity()
        system_config.ac_capacity = system_config.calculate_ac_capacity()
        system_config.dc_ac_ratio = system_config.calculate_dc_ac_ratio()

        logger.info(
            f"System design complete: {system_config.dc_capacity:.1f}kW DC, "
            f"{system_config.ac_capacity:.1f}kW AC, "
            f"DC/AC={system_config.dc_ac_ratio:.2f}, "
            f"{num_modules} modules, {num_inverters} inverters"
        )

        return system_config

    def calculate_string_sizing(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        site_temp_min: float = -10.0,
        site_temp_max: float = 70.0,
    ) -> StringConfiguration:
        """
        Calculate optimal string sizing.

        Args:
            module: Module parameters
            inverter: Inverter parameters
            site_temp_min: Minimum site temperature (°C)
            site_temp_max: Maximum site temperature (°C)

        Returns:
            Optimized StringConfiguration
        """
        logger.info("Calculating string sizing")

        calculator = StringSizingCalculator(
            module=module,
            inverter=inverter,
            site_temp_min=site_temp_min,
            site_temp_max=site_temp_max,
        )

        # Get optimal string configuration
        string_config = calculator.design_optimal_string()

        logger.info(
            f"String sizing: {string_config.modules_per_string} modules/string, "
            f"{string_config.strings_per_mppt} strings/MPPT"
        )

        return string_config

    def select_inverters(
        self,
        module: ModuleParameters,
        dc_capacity_kw: float,
        target_dc_ac_ratio: float = 1.25,
        inverter_database_path: Optional[str] = None,
    ) -> List[InverterParameters]:
        """
        Select suitable inverters from database.

        Args:
            module: Module parameters
            dc_capacity_kw: Total DC capacity in kW
            target_dc_ac_ratio: Target DC/AC ratio
            inverter_database_path: Path to inverter database

        Returns:
            List of suitable InverterParameters
        """
        logger.info(f"Selecting inverters for {dc_capacity_kw}kW DC capacity")

        selector = InverterSelector(
            module=module,
            system_type=self.system_type,
            database_path=inverter_database_path,
        )

        # Search for suitable inverters
        candidates = selector.search_inverter_database(
            dc_power_kw=dc_capacity_kw,
        )

        if not candidates:
            logger.warning("No inverters found in database")
            return []

        logger.info(f"Found {len(candidates)} suitable inverters")

        return candidates[:5]  # Return top 5 candidates

    def design_dc_wiring(
        self,
        array_layout: ArrayLayout,
        string_config: StringConfiguration,
        cable_cross_section_mm2: float = 6.0,
    ) -> Dict[str, float]:
        """
        Design DC wiring and calculate cable requirements.

        Args:
            array_layout: Array layout configuration
            string_config: String configuration
            cable_cross_section_mm2: DC cable cross-section (mm²)

        Returns:
            Dictionary with DC wiring design parameters
        """
        logger.info("Designing DC wiring system")

        # Use layout designer for string routing
        if not self.layout_designer:
            raise ValueError("Layout designer not initialized")

        routing = self.layout_designer.design_string_routing(
            layout=array_layout,
            modules_per_string=string_config.modules_per_string,
        )

        # Calculate voltage drop
        if self.loss_model:
            cable_length_m = routing['avg_string_cable_length_m']
            current_a = string_config.imp_stc or 10.0

            voltage_drop_percent = self.loss_model.dc_wiring_losses(
                cable_length_m=cable_length_m,
                cable_cross_section_mm2=cable_cross_section_mm2,
                current_a=current_a,
            )

            routing['cable_cross_section_mm2'] = cable_cross_section_mm2
            routing['voltage_drop_percent'] = voltage_drop_percent

        logger.info(
            f"DC wiring: {routing['total_dc_cable_estimate_m']:.0f}m total cable, "
            f"{routing['num_combiners']} combiners"
        )

        return routing

    def design_ac_collection(
        self,
        num_inverters: int,
        inverter_ac_power_kw: float,
        distance_to_poc_m: float = 500.0,
    ) -> Dict[str, any]:
        """
        Design AC collection system and transformer sizing.

        Args:
            num_inverters: Number of inverters
            inverter_ac_power_kw: AC power per inverter (kW)
            distance_to_poc_m: Distance to point of connection (m)

        Returns:
            Dictionary with AC collection system design
        """
        logger.info("Designing AC collection system")

        total_ac_power_kw = num_inverters * inverter_ac_power_kw

        # Transformer sizing (110% of AC capacity)
        transformer_kva = total_ac_power_kw * 1.1

        # Standard transformer sizes (kVA)
        standard_sizes = [
            500, 750, 1000, 1500, 2000, 2500, 3000, 5000, 7500, 10000, 15000, 20000
        ]
        transformer_size_kva = next(
            (size for size in standard_sizes if size >= transformer_kva),
            transformer_kva
        )

        # AC collection voltage (based on system size)
        if self.system_type == SystemType.RESIDENTIAL:
            collection_voltage_v = 240
            num_phases = 1
        elif self.system_type == SystemType.COMMERCIAL:
            collection_voltage_v = 480
            num_phases = 3
        else:  # UTILITY
            collection_voltage_v = 34500 if total_ac_power_kw > 5000 else 12470
            num_phases = 3

        ac_design = {
            'total_ac_power_kw': total_ac_power_kw,
            'transformer_size_kva': transformer_size_kva,
            'collection_voltage_v': collection_voltage_v,
            'num_phases': num_phases,
            'distance_to_poc_m': distance_to_poc_m,
            'num_inverters': num_inverters,
        }

        # Calculate AC wiring losses if loss model available
        if self.loss_model:
            ac_loss_percent = self.loss_model.ac_wiring_losses(
                cable_length_m=distance_to_poc_m,
                cable_cross_section_mm2=50.0,  # Typical for utility
                power_kw=total_ac_power_kw,
                voltage_v=collection_voltage_v,
                num_phases=num_phases,
            )
            ac_design['ac_wiring_loss_percent'] = ac_loss_percent

        logger.info(
            f"AC collection: {transformer_size_kva}kVA transformer, "
            f"{collection_voltage_v}V {num_phases}-phase"
        )

        return ac_design

    def calculate_system_losses(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        mounting_type: MountingType,
        **kwargs,
    ) -> SystemLosses:
        """
        Calculate comprehensive system losses.

        Args:
            module: Module parameters
            inverter: Inverter parameters
            mounting_type: Mounting type
            **kwargs: Additional loss parameters

        Returns:
            SystemLosses object
        """
        logger.info("Calculating system losses")

        loss_model = SystemLossModel(
            module=module,
            inverter=inverter,
            mounting_type=mounting_type,
        )

        # Calculate losses at nominal operating point
        dc_power_w = module.pmax

        losses = loss_model.calculate_system_losses(
            dc_power_w=dc_power_w,
            location_type=kwargs.get('location_type', 'temperate'),
            cable_length_dc_m=kwargs.get('cable_length_dc_m', 100.0),
            cable_length_ac_m=kwargs.get('cable_length_ac_m', 200.0),
            has_transformer=kwargs.get('has_transformer', self.system_type == SystemType.UTILITY),
            **kwargs,
        )

        logger.info(f"Total system losses: {losses.total_losses():.2f}%")

        return losses

    def design_array_layout(
        self,
        total_modules: int,
        mounting_type: MountingType,
        modules_per_string: int,
        **kwargs,
    ) -> ArrayLayout:
        """
        Design array layout based on mounting type.

        Args:
            total_modules: Total number of modules
            mounting_type: Type of mounting system
            modules_per_string: Modules per string
            **kwargs: Additional layout parameters

        Returns:
            ArrayLayout configuration
        """
        logger.info(f"Designing {mounting_type.value} array layout for {total_modules} modules")

        if not self.layout_designer:
            raise ValueError("Layout designer not initialized")

        # Route to appropriate layout method
        if mounting_type in [
            MountingType.GROUND_FIXED,
            MountingType.GROUND_SINGLE_AXIS,
            MountingType.GROUND_DUAL_AXIS,
        ]:
            tracker_type = (
                "single_axis" if mounting_type == MountingType.GROUND_SINGLE_AXIS
                else "dual_axis" if mounting_type == MountingType.GROUND_DUAL_AXIS
                else "fixed"
            )
            layout = self.layout_designer.ground_mount_layout(
                total_modules=total_modules,
                target_gcr=kwargs.get('target_gcr', 0.4),
                tracker_type=tracker_type,
                modules_per_row=kwargs.get('modules_per_row', modules_per_string),
            )

        elif mounting_type in [MountingType.ROOFTOP_FLAT, MountingType.ROOFTOP_SLOPED]:
            layout = self.layout_designer.rooftop_layout(
                total_modules=total_modules,
                roof_width_m=kwargs.get('roof_width_m', 50.0),
                roof_length_m=kwargs.get('roof_length_m', 100.0),
                roof_tilt=kwargs.get('roof_tilt', 5.0 if mounting_type == MountingType.ROOFTOP_FLAT else 20.0),
            )

        elif mounting_type == MountingType.CARPORT:
            layout = self.layout_designer.carport_canopy_layout(
                total_modules=total_modules,
                num_parking_spaces=kwargs.get('num_parking_spaces', 100),
            )

        elif mounting_type == MountingType.FLOATING:
            layout = self.layout_designer.floating_pv_layout(
                total_modules=total_modules,
                water_area_m2=kwargs.get('water_area_m2', 50000.0),
            )

        elif mounting_type == MountingType.AGRIVOLTAIC:
            layout = self.layout_designer.agrivoltaic_layout(
                total_modules=total_modules,
                field_area_m2=kwargs.get('field_area_m2', 100000.0),
            )

        else:
            # Default to ground fixed
            layout = self.layout_designer.ground_mount_layout(
                total_modules=total_modules,
                modules_per_row=modules_per_string,
            )

        logger.info(f"Array layout: {layout.rows} rows × {layout.modules_per_row} modules")

        return layout

    def optimize_system_layout(
        self,
        module: ModuleParameters,
        inverter: InverterParameters,
        available_area_m2: float,
        mounting_type: MountingType = MountingType.GROUND_FIXED,
        **kwargs,
    ) -> SystemConfiguration:
        """
        Optimize system layout for maximum energy yield within available area.

        Args:
            module: Module parameters
            inverter: Inverter parameters
            available_area_m2: Available area in m²
            mounting_type: Type of mounting system
            **kwargs: Additional optimization parameters

        Returns:
            Optimized SystemConfiguration
        """
        logger.info(f"Optimizing system layout for {available_area_m2:.0f}m² area")

        # Initialize layout designer
        layout_designer = ArrayLayoutDesigner(
            module=module,
            mounting_type=mounting_type,
            site_latitude=self.latitude,
        )

        # Determine optimal tilt and spacing
        tilt_angle = kwargs.get('tilt_angle', abs(self.latitude))

        if mounting_type in [MountingType.GROUND_FIXED, MountingType.GROUND_SINGLE_AXIS]:
            # Optimize row spacing
            optimal_spacing = layout_designer.optimize_row_spacing(
                tilt_angle=tilt_angle,
                max_shading_loss=kwargs.get('max_shading_loss', 5.0),
            )

            # Calculate GCR
            gcr = layout_designer.calculate_ground_coverage_ratio(
                tilt_angle=tilt_angle,
                row_spacing=optimal_spacing,
            )

            # Calculate how many modules fit
            module_area = module.area
            usable_modules = int((available_area_m2 * gcr) / module_area)

        else:
            # For other mounting types, use simpler calculation
            module_area = module.area
            usable_modules = int((available_area_m2 * 0.7) / module_area)  # 70% utilization
            gcr = 0.7

        # Calculate DC capacity
        dc_capacity_kw = (usable_modules * module.pmax) / 1000

        # Design complete system
        system_config = self.design_system_configuration(
            module=module,
            inverter=inverter,
            target_dc_capacity_kw=dc_capacity_kw,
            mounting_type=mounting_type,
            **kwargs,
        )

        logger.info(
            f"Optimized layout: {usable_modules} modules, "
            f"{dc_capacity_kw:.1f}kW DC, GCR={gcr:.2f}"
        )

        return system_config
