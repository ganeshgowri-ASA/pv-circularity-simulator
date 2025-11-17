"""
Export functionality for PV Circularity visualizations.

This module provides comprehensive export capabilities for visualizations,
supporting multiple formats including static images, interactive HTML,
vector graphics, and data exports.
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import io


class ExportManager:
    """
    Manages export operations for visualizations.

    This class handles exporting visualizations to various formats including
    PNG, JPEG, SVG, PDF, HTML, and JSON. It supports both Plotly and Altair
    visualizations with configurable quality and size settings.

    Examples:
        >>> exporter = ExportManager()
        >>> exporter.export_plotly(fig, 'output.png', width=1200, height=800)
        >>> exporter.export_to_html(fig, 'interactive.html', include_plotlyjs=True)
    """

    def __init__(self) -> None:
        """Initialize the export manager with default settings."""
        self.default_width: int = 1200
        self.default_height: int = 800
        self.default_scale: float = 2.0  # For high-DPI displays
        self.default_quality: int = 95  # For JPEG exports

    def export_plotly(
        self,
        fig: go.Figure,
        filepath: Union[str, Path],
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: Optional[float] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Export a Plotly figure to a file.

        Args:
            fig: Plotly figure to export
            filepath: Output file path
            format: Export format (auto-detected from extension if None)
            width: Width in pixels (uses default if None)
            height: Height in pixels (uses default if None)
            scale: Scale factor for image quality (uses default if None)
            **kwargs: Additional format-specific parameters

        Returns:
            Path to the exported file

        Raises:
            ValueError: If format is unsupported
            IOError: If export fails

        Examples:
            >>> exporter = ExportManager()
            >>> fig = go.Figure(data=[go.Scatter(x=[1,2,3], y=[4,5,6])])
            >>> exporter.export_plotly(fig, 'chart.png', width=1600, height=900)
        """
        filepath = Path(filepath)
        format = format or filepath.suffix.lstrip(".")

        width = width or self.default_width
        height = height or self.default_height
        scale = scale or self.default_scale

        # Validate format
        supported_formats = ["png", "jpg", "jpeg", "svg", "pdf", "html", "json"]
        if format.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Export based on format
        if format.lower() in ["png", "jpg", "jpeg", "svg", "pdf"]:
            self._export_static_image(
                fig, filepath, format, width, height, scale, **kwargs
            )
        elif format.lower() == "html":
            self._export_html(fig, filepath, **kwargs)
        elif format.lower() == "json":
            self._export_json(fig, filepath, **kwargs)

        return filepath

    def _export_static_image(
        self,
        fig: go.Figure,
        filepath: Path,
        format: str,
        width: int,
        height: int,
        scale: float,
        **kwargs: Any,
    ) -> None:
        """
        Export figure as a static image.

        Args:
            fig: Plotly figure to export
            filepath: Output file path
            format: Image format (png, jpg, jpeg, svg, pdf)
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for quality
            **kwargs: Additional parameters
        """
        try:
            # Handle JPEG quality parameter
            if format.lower() in ["jpg", "jpeg"]:
                kwargs.setdefault("quality", self.default_quality)

            # Export using kaleido
            pio.write_image(
                fig,
                str(filepath),
                format=format,
                width=width,
                height=height,
                scale=scale,
                **kwargs,
            )
        except Exception as e:
            raise IOError(f"Failed to export image: {str(e)}") from e

    def _export_html(
        self,
        fig: go.Figure,
        filepath: Path,
        include_plotlyjs: Union[bool, str] = "cdn",
        auto_open: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Export figure as interactive HTML.

        Args:
            fig: Plotly figure to export
            filepath: Output file path
            include_plotlyjs: How to include Plotly.js ('cdn', True, False)
            auto_open: Whether to open in browser after export
            **kwargs: Additional parameters
        """
        try:
            fig.write_html(
                str(filepath),
                include_plotlyjs=include_plotlyjs,
                auto_open=auto_open,
                **kwargs,
            )
        except Exception as e:
            raise IOError(f"Failed to export HTML: {str(e)}") from e

    def _export_json(
        self,
        fig: go.Figure,
        filepath: Path,
        pretty: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Export figure as JSON.

        Args:
            fig: Plotly figure to export
            filepath: Output file path
            pretty: Whether to pretty-print JSON
            **kwargs: Additional parameters
        """
        try:
            fig_dict = fig.to_dict()
            with open(filepath, "w") as f:
                if pretty:
                    json.dump(fig_dict, f, indent=2, **kwargs)
                else:
                    json.dump(fig_dict, f, **kwargs)
        except Exception as e:
            raise IOError(f"Failed to export JSON: {str(e)}") from e

    def export_altair(
        self,
        chart: Any,  # altair.Chart
        filepath: Union[str, Path],
        format: Optional[str] = None,
        scale_factor: float = 2.0,
        **kwargs: Any,
    ) -> Path:
        """
        Export an Altair chart to a file.

        Args:
            chart: Altair chart to export
            filepath: Output file path
            format: Export format (auto-detected from extension if None)
            scale_factor: Scale factor for image quality
            **kwargs: Additional format-specific parameters

        Returns:
            Path to the exported file

        Raises:
            ValueError: If format is unsupported
            IOError: If export fails

        Examples:
            >>> import altair as alt
            >>> exporter = ExportManager()
            >>> chart = alt.Chart(data).mark_bar().encode(x='x', y='y')
            >>> exporter.export_altair(chart, 'chart.png')
        """
        filepath = Path(filepath)
        format = format or filepath.suffix.lstrip(".")

        # Validate format
        supported_formats = ["png", "svg", "html", "json"]
        if format.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Create parent directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format.lower() == "png":
                chart.save(str(filepath), scale_factor=scale_factor, **kwargs)
            elif format.lower() == "svg":
                chart.save(str(filepath), **kwargs)
            elif format.lower() == "html":
                chart.save(str(filepath), **kwargs)
            elif format.lower() == "json":
                with open(filepath, "w") as f:
                    f.write(chart.to_json(**kwargs))
        except Exception as e:
            raise IOError(f"Failed to export Altair chart: {str(e)}") from e

        return filepath

    def export_multiple(
        self,
        figures: Dict[str, go.Figure],
        output_dir: Union[str, Path],
        format: str = "png",
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Path]:
        """
        Export multiple figures to files.

        Args:
            figures: Dictionary mapping filenames to figures
            output_dir: Directory to save exported files
            format: Export format for all figures
            width: Width in pixels (uses default if None)
            height: Height in pixels (uses default if None)
            **kwargs: Additional export parameters

        Returns:
            List of paths to exported files

        Examples:
            >>> exporter = ExportManager()
            >>> figures = {
            ...     'chart1': fig1,
            ...     'chart2': fig2,
            ...     'chart3': fig3
            ... }
            >>> paths = exporter.export_multiple(figures, 'output/', format='png')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        for filename, fig in figures.items():
            # Add extension if not present
            if not filename.endswith(f".{format}"):
                filename = f"{filename}.{format}"

            filepath = output_dir / filename
            self.export_plotly(
                fig, filepath, format=format, width=width, height=height, **kwargs
            )
            exported_paths.append(filepath)

        return exported_paths

    def create_image_grid(
        self,
        figures: List[go.Figure],
        output_path: Union[str, Path],
        rows: int,
        cols: int,
        width_per_cell: int = 600,
        height_per_cell: int = 400,
        padding: int = 10,
        background: str = "#FFFFFF",
    ) -> Path:
        """
        Create a grid of multiple figures in a single image.

        Args:
            figures: List of Plotly figures to arrange in grid
            output_path: Path to save the combined image
            rows: Number of rows in grid
            cols: Number of columns in grid
            width_per_cell: Width of each cell in pixels
            height_per_cell: Height of each cell in pixels
            padding: Padding between cells in pixels
            background: Background color for the grid

        Returns:
            Path to the exported grid image

        Raises:
            ValueError: If number of figures exceeds grid capacity
            IOError: If export fails

        Examples:
            >>> exporter = ExportManager()
            >>> figures = [fig1, fig2, fig3, fig4]
            >>> exporter.create_image_grid(figures, 'grid.png', rows=2, cols=2)
        """
        if len(figures) > rows * cols:
            raise ValueError(
                f"Too many figures ({len(figures)}) for {rows}x{cols} grid"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate total dimensions
        total_width = cols * width_per_cell + (cols + 1) * padding
        total_height = rows * height_per_cell + (rows + 1) * padding

        # Create blank canvas
        canvas = Image.new("RGB", (total_width, total_height), background)

        # Export each figure and paste into grid
        for idx, fig in enumerate(figures):
            row = idx // cols
            col = idx % cols

            # Export figure to bytes
            img_bytes = pio.to_image(
                fig,
                format="png",
                width=width_per_cell,
                height=height_per_cell,
                scale=2.0,
            )
            img = Image.open(io.BytesIO(img_bytes))

            # Calculate position
            x = col * width_per_cell + (col + 1) * padding
            y = row * height_per_cell + (row + 1) * padding

            # Paste image onto canvas
            canvas.paste(img, (x, y))

        # Save combined image
        canvas.save(output_path, quality=self.default_quality)

        return output_path

    def export_with_data(
        self,
        fig: go.Figure,
        filepath: Union[str, Path],
        data: Any,
        data_format: str = "csv",
        **kwargs: Any,
    ) -> Dict[str, Path]:
        """
        Export visualization along with its underlying data.

        Args:
            fig: Plotly figure to export
            filepath: Base path for exports (extension will be modified)
            data: Data to export (DataFrame, dict, etc.)
            data_format: Format for data export ('csv', 'json', 'excel')
            **kwargs: Additional export parameters

        Returns:
            Dictionary with 'figure' and 'data' file paths

        Examples:
            >>> import pandas as pd
            >>> exporter = ExportManager()
            >>> df = pd.DataFrame({'x': [1,2,3], 'y': [4,5,6]})
            >>> fig = go.Figure(data=[go.Scatter(x=df['x'], y=df['y'])])
            >>> paths = exporter.export_with_data(fig, 'chart', df, 'csv')
        """
        filepath = Path(filepath)
        base_path = filepath.parent / filepath.stem

        # Export figure
        fig_path = self.export_plotly(fig, f"{base_path}.png", **kwargs)

        # Export data
        data_path = base_path.with_suffix(f".{data_format}")

        if hasattr(data, "to_csv") and data_format == "csv":
            data.to_csv(data_path, index=False)
        elif hasattr(data, "to_excel") and data_format == "excel":
            data.to_excel(data_path, index=False)
        elif hasattr(data, "to_json") and data_format == "json":
            data.to_json(data_path, orient="records", indent=2)
        elif data_format == "json":
            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        return {"figure": fig_path, "data": data_path}


# Global export manager instance
_global_export_manager = ExportManager()


def get_export_manager() -> ExportManager:
    """
    Get the global export manager instance.

    Returns:
        Global ExportManager instance

    Examples:
        >>> from pv_circularity.visualization.exports import get_export_manager
        >>> exporter = get_export_manager()
        >>> exporter.export_plotly(fig, 'output.png')
    """
    return _global_export_manager


def export_functionality() -> ExportManager:
    """
    Access the export functionality system.

    This is a convenience function that returns the global export manager,
    providing access to all export capabilities.

    Returns:
        ExportManager instance for handling exports

    Examples:
        >>> from pv_circularity.visualization import export_functionality
        >>> exporter = export_functionality()
        >>> exporter.export_plotly(fig, 'chart.png', width=1600)
    """
    return get_export_manager()
