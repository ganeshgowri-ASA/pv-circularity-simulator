"""PAN file generator for PVsyst compatibility."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from ..models.module import ModuleDesign


class PANFileContent(BaseModel):
    """PAN file content structure."""

    content: str = Field(..., description="PAN file content")
    filename: str = Field(..., description="Suggested filename")
    module_name: str = Field(..., description="Module name")


class PANGenerator:
    """Generator for PVsyst PAN files."""

    def __init__(self, module: ModuleDesign):
        """Initialize PAN generator.

        Args:
            module: Module design specification
        """
        self.module = module

    def generate(
        self,
        module_name: str,
        manufacturer: str = "Custom Manufacturer",
        model_number: str = "CUSTOM-001",
        date: Optional[str] = None,
    ) -> PANFileContent:
        """Generate PAN file content.

        Args:
            module_name: Module name
            manufacturer: Manufacturer name
            model_number: Model number
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            PANFileContent with PAN file data
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Calculate module parameters
        num_cells = self.module.configuration.layout.num_cells
        cells_series = self.module.configuration.layout.cells_per_string
        cells_parallel = self.module.configuration.layout.num_strings

        # Technology type mapping
        tech_map = {
            "mono-Si": "mtSiMono",
            "multi-Si": "mtSiPoly",
            "PERC": "mtSiMono",
            "TOPCon": "mtSiMono",
            "HJT": "mtHIT",
            "IBC": "mtSiMono",
            "Perovskite": "mtThinFilm",
            "Tandem": "mtThinFilm",
        }
        technology = tech_map.get(self.module.configuration.cell_design.template.technology, "mtSiMono")

        # Generate PAN content
        pan_content = f"""PVObject_=pvModule
  Version=7.2
  Flags=$0043

  PVObject_Commercial=pvCommercial
    Comment={module_name}
    Manufacturer={manufacturer}
    Model={model_number}
    DataSource=User-defined
    YearBeg={date[:4]}
    Width={self.module.configuration.width_mm / 1000:.3f}
    Height={self.module.configuration.length_mm / 1000:.3f}
    Depth={self.module.configuration.thickness_mm / 1000:.3f}
    Weight={self.module.configuration.weight_kg or 0:.1f}
    NPieces=100
    PriceDate={date}
  End of PVObject pvCommercial

  Technol={technology}
  NCelS={cells_series}
  NCelP={cells_parallel}
  NDiode={self.module.configuration.layout.bypass_diodes}
  FrontSurface=fsARCoating
  {"GlassThickness=" + f"{self.module.configuration.glass_front_mm:.1f}" if self.module.configuration.glass_front_mm else ""}
  SubModule=smStandard
  Tolerance={3.0}
  {"Bifacial=Yes" if self.module.configuration.is_bifacial else ""}
  {"BifacialityFactor=" + f"{self.module.configuration.cell_design.template.bifaciality_factor * 100:.1f}" if self.module.configuration.is_bifacial else ""}

  // STC Performance Parameters
  Pnom={self.module.pmax_stc_w:.2f}
  Voc={self.module.voc_stc_v:.3f}
  Isc={self.module.isc_stc_a:.3f}
  Vmp={self.module.vmp_stc_v:.3f}
  Imp={self.module.imp_stc_a:.3f}
  muISC={self.module.temp_coeff_isc_pct:.4f}
  muVocSpec={self.module.temp_coeff_voc_pct / 100:.6f}
  muPmpReq={self.module.temp_coeff_pmax_pct:.4f}

  // NOCT Parameters
  NOCTAmbient=20
  NOCT={self.module.noct_temp_c:.1f}
  PNom_NOCT={self.module.pmax_noct_w:.2f}

  // Operating Conditions
  TempCoef={self.module.temp_coeff_pmax_pct:.4f}
  TempCoefVoc={self.module.temp_coeff_voc_pct / 100:.6f}
  TempCoefIsc={self.module.temp_coeff_isc_pct:.4f}

  // Diode and Series Resistance Parameters
  VMaxIEC={int(self.module.max_system_voltage_v)}
  VMaxUL={int(self.module.max_system_voltage_v)}
  Imax={self.module.max_series_fuse_a:.1f}

  // Module Quality and Degradation
  IamMode=ashrae
  IAM_c_as=0.05

  // One-diode model parameters (approximated)
  Rshunt={1000 + (num_cells * 10):.0f}
  Rp_0={2000 + (num_cells * 20):.0f}
  Rp_Exp=5.50
  RSerie={0.200 + (cells_series * 0.005):.3f}
  Gamma={1.100:.3f}
  muGamma={-0.0005:.6f}

  // Cell parameters
  NCelSer={cells_series}
  NCelPar={cells_parallel}

  // Estimated cell efficiency
  CellArea={(self.module.configuration.cell_design.template.length_mm * self.module.configuration.cell_design.template.width_mm) / 1_000_000:.6f}

  // Module efficiency
  AreaModuleEff={self.module.configuration.area_m2:.4f}

End of PVObject pvModule
"""

        filename = f"{manufacturer}_{model_number}_{int(self.module.pmax_stc_w)}W.PAN"
        filename = filename.replace(" ", "_")

        return PANFileContent(content=pan_content, filename=filename, module_name=module_name)

    def validate(self, pan_content: str) -> dict:
        """Validate PAN file content structure.

        Args:
            pan_content: PAN file content to validate

        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []

        # Check for required fields
        required_fields = [
            "Pnom=",
            "Voc=",
            "Isc=",
            "Vmp=",
            "Imp=",
            "NCelS=",
            "NCelP=",
            "muISC=",
            "muVocSpec=",
            "muPmpReq=",
        ]

        for field in required_fields:
            if field not in pan_content:
                errors.append(f"Missing required field: {field}")

        # Check PVObject structure
        if not pan_content.startswith("PVObject_=pvModule"):
            errors.append("PAN file must start with 'PVObject_=pvModule'")

        if "End of PVObject pvModule" not in pan_content:
            errors.append("PAN file must end with 'End of PVObject pvModule'")

        # Check commercial section
        if "PVObject_Commercial=pvCommercial" not in pan_content:
            warnings.append("Missing commercial information section")

        # Validate power consistency
        try:
            pnom_line = [line for line in pan_content.split("\n") if "Pnom=" in line and "PNom" not in line][0]
            pnom = float(pnom_line.split("=")[1].strip())

            vmp_line = [line for line in pan_content.split("\n") if "Vmp=" in line][0]
            vmp = float(vmp_line.split("=")[1].strip())

            imp_line = [line for line in pan_content.split("\n") if "Imp=" in line][0]
            imp = float(imp_line.split("=")[1].strip())

            calculated_pnom = vmp * imp

            if abs(calculated_pnom - pnom) > 0.5:
                warnings.append(
                    f"Power inconsistency: Pnom={pnom:.2f}W but Vmp×Imp={calculated_pnom:.2f}W"
                )
        except (IndexError, ValueError):
            warnings.append("Could not validate power consistency")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
