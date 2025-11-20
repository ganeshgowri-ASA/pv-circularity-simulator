"""Pre-defined cell templates."""

from typing import Dict, List
from ..models.cell import CellTemplate


def get_cell_templates() -> Dict[str, CellTemplate]:
    """Get dictionary of pre-defined cell templates.

    Returns:
        Dictionary mapping template name to CellTemplate
    """
    return {
        "M10_PERC_Mono": CellTemplate(
            name="M10 PERC Monocrystalline",
            technology="PERC",
            cell_type="M10",
            length_mm=182.0,
            width_mm=182.0,
            thickness_um=180,
            efficiency_pct=23.0,
            pmax_w=6.1,
            voc_v=0.710,
            isc_a=11.5,
            vmp_v=0.600,
            imp_a=10.2,
            fill_factor=0.794,
            temp_coeff_pmax=-0.35,
            temp_coeff_voc=-0.28,
            temp_coeff_isc=0.05,
            bifacial=False,
        ),
        "M10_TOPCon_Mono_Bifacial": CellTemplate(
            name="M10 TOPCon Monocrystalline Bifacial",
            technology="TOPCon",
            cell_type="M10",
            length_mm=182.0,
            width_mm=182.0,
            thickness_um=160,
            efficiency_pct=24.5,
            pmax_w=6.5,
            voc_v=0.730,
            isc_a=11.8,
            vmp_v=0.620,
            imp_a=10.5,
            fill_factor=0.813,
            temp_coeff_pmax=-0.29,
            temp_coeff_voc=-0.25,
            temp_coeff_isc=0.04,
            bifacial=True,
            bifaciality_factor=0.75,
        ),
        "M12_TOPCon_Mono_Bifacial": CellTemplate(
            name="M12 TOPCon Monocrystalline Bifacial",
            technology="TOPCon",
            cell_type="M12",
            length_mm=210.0,
            width_mm=210.0,
            thickness_um=160,
            efficiency_pct=24.8,
            pmax_w=8.7,
            voc_v=0.740,
            isc_a=15.6,
            vmp_v=0.630,
            imp_a=13.8,
            fill_factor=0.820,
            temp_coeff_pmax=-0.28,
            temp_coeff_voc=-0.24,
            temp_coeff_isc=0.04,
            bifacial=True,
            bifaciality_factor=0.78,
        ),
        "M6_PERC_Mono": CellTemplate(
            name="M6 PERC Monocrystalline",
            technology="PERC",
            cell_type="M6",
            length_mm=166.0,
            width_mm=166.0,
            thickness_um=180,
            efficiency_pct=22.5,
            pmax_w=5.2,
            voc_v=0.695,
            isc_a=10.0,
            vmp_v=0.585,
            imp_a=8.9,
            fill_factor=0.789,
            temp_coeff_pmax=-0.37,
            temp_coeff_voc=-0.29,
            temp_coeff_isc=0.05,
            bifacial=False,
        ),
        "M10_HJT_Bifacial": CellTemplate(
            name="M10 HJT (Heterojunction) Bifacial",
            technology="HJT",
            cell_type="M10",
            length_mm=182.0,
            width_mm=182.0,
            thickness_um=120,
            efficiency_pct=25.2,
            pmax_w=6.8,
            voc_v=0.750,
            isc_a=12.1,
            vmp_v=0.640,
            imp_a=10.6,
            fill_factor=0.835,
            temp_coeff_pmax=-0.24,
            temp_coeff_voc=-0.22,
            temp_coeff_isc=0.03,
            bifacial=True,
            bifaciality_factor=0.92,
        ),
        "M10_IBC_Mono": CellTemplate(
            name="M10 IBC (Interdigitated Back Contact)",
            technology="IBC",
            cell_type="M10",
            length_mm=182.0,
            width_mm=182.0,
            thickness_um=160,
            efficiency_pct=25.5,
            pmax_w=6.9,
            voc_v=0.755,
            isc_a=12.2,
            vmp_v=0.650,
            imp_a=10.6,
            fill_factor=0.842,
            temp_coeff_pmax=-0.25,
            temp_coeff_voc=-0.23,
            temp_coeff_isc=0.04,
            bifacial=False,
        ),
        "G12_PERC_Multi": CellTemplate(
            name="G12 PERC Multicrystalline",
            technology="PERC",
            cell_type="G12",
            length_mm=210.0,
            width_mm=210.0,
            thickness_um=200,
            efficiency_pct=21.5,
            pmax_w=7.5,
            voc_v=0.680,
            isc_a=14.8,
            vmp_v=0.570,
            imp_a=13.2,
            fill_factor=0.765,
            temp_coeff_pmax=-0.40,
            temp_coeff_voc=-0.31,
            temp_coeff_isc=0.06,
            bifacial=False,
        ),
    }


def get_cell_template_by_name(name: str) -> CellTemplate:
    """Get cell template by name.

    Args:
        name: Template name

    Returns:
        CellTemplate

    Raises:
        KeyError: If template name not found
    """
    templates = get_cell_templates()
    return templates[name]


def get_template_names() -> List[str]:
    """Get list of available template names.

    Returns:
        List of template names
    """
    return list(get_cell_templates().keys())
