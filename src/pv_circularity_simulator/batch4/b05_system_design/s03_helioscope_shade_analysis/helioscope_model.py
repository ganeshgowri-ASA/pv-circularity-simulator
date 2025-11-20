"""
Helioscope 3D Model for site terrain, obstacles, and PV system visualization.

This module provides comprehensive 3D modeling capabilities including terrain
modeling, obstacle import, and system visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    ArrayGeometry,
    FileFormat,
    Location,
    Obstacle,
    SiteModel,
    TerrainData,
)

logger = logging.getLogger(__name__)


class HelioscapeModel:
    """
    3D site model for PV system design and shade analysis.

    This class handles import of 3D models, terrain modeling, obstacle
    definition, and 3D visualization of PV systems on terrain.
    """

    def __init__(self, site_model: SiteModel):
        """
        Initialize the Helioscope model.

        Args:
            site_model: Site model with location and terrain data
        """
        self.site_model = site_model
        self.pv_arrays: List[Dict[str, Any]] = []
        self._3d_mesh_cache: Dict[str, Any] = {}

        logger.info(
            f"Initialized HelioscapeModel for location: "
            f"({site_model.location.latitude:.4f}, {site_model.location.longitude:.4f})"
        )

    def import_3d_model(
        self,
        file_path: Path,
        file_format: Optional[FileFormat] = None,
        scale_factor: float = 1.0,
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> Obstacle:
        """
        Import 3D model from file (SketchUp, OBJ, STL, DXF).

        Args:
            file_path: Path to 3D model file
            file_format: File format (auto-detected from extension if None)
            scale_factor: Scale factor to apply to model
            offset: (x, y, z) offset to apply to model coordinates

        Returns:
            Obstacle object containing the imported 3D model

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"3D model file not found: {file_path}")

        # Auto-detect format from extension
        if file_format is None:
            extension = file_path.suffix.lower().lstrip('.')
            try:
                file_format = FileFormat(extension)
            except ValueError:
                raise ValueError(
                    f"Unsupported file format: {extension}. "
                    f"Supported formats: {[f.value for f in FileFormat]}"
                )

        logger.info(f"Importing 3D model from {file_path} (format: {file_format.value})")

        # Import based on format
        if file_format == FileFormat.OBJ:
            vertices, faces = self._import_obj(file_path)
        elif file_format == FileFormat.STL:
            vertices, faces = self._import_stl(file_path)
        elif file_format == FileFormat.DXF:
            vertices, faces = self._import_dxf(file_path)
        elif file_format == FileFormat.PLY:
            vertices, faces = self._import_ply(file_path)
        else:
            # SketchUp requires specialized library
            vertices, faces = self._import_sketchup(file_path)

        # Apply transformations
        vertices = self._apply_transformations(vertices, scale_factor, offset)

        # Calculate bounding box and height
        heights = [v[2] for v in vertices]
        max_height = max(heights) - min(heights)

        obstacle = Obstacle(
            name=file_path.stem,
            vertices=vertices,
            faces=faces,
            height=max_height,
            obstacle_type="imported_model"
        )

        self.site_model.obstacles.append(obstacle)
        logger.info(f"Imported 3D model with {len(vertices)} vertices and {len(faces)} faces")

        return obstacle

    def terrain_modeling(
        self,
        survey_data: Optional[pd.DataFrame] = None,
        resolution: float = 1.0,
        interpolation_method: str = "linear"
    ) -> TerrainData:
        """
        Create digital terrain model (DTM) from survey data or satellite imagery.

        Args:
            survey_data: DataFrame with columns ['x', 'y', 'elevation']
            resolution: Grid resolution in meters
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')

        Returns:
            TerrainData object with interpolated terrain grid

        Raises:
            ValueError: If survey data is invalid
        """
        if survey_data is None:
            # Generate flat terrain as default
            logger.warning("No survey data provided, generating flat terrain")
            x_coords = np.arange(0, 100, resolution).tolist()
            y_coords = np.arange(0, 100, resolution).tolist()
            elevations = [[0.0 for _ in x_coords] for _ in y_coords]

            terrain_data = TerrainData(
                x_coordinates=x_coords,
                y_coordinates=y_coords,
                elevations=elevations,
                resolution=resolution
            )

        else:
            # Validate survey data
            required_columns = ['x', 'y', 'elevation']
            if not all(col in survey_data.columns for col in required_columns):
                raise ValueError(f"Survey data must contain columns: {required_columns}")

            # Create regular grid
            x_min, x_max = survey_data['x'].min(), survey_data['x'].max()
            y_min, y_max = survey_data['y'].min(), survey_data['y'].max()

            x_coords = np.arange(x_min, x_max + resolution, resolution)
            y_coords = np.arange(y_min, y_max + resolution, resolution)

            # Interpolate elevations
            elevations = self._interpolate_terrain(
                survey_data,
                x_coords,
                y_coords,
                interpolation_method
            )

            terrain_data = TerrainData(
                x_coordinates=x_coords.tolist(),
                y_coordinates=y_coords.tolist(),
                elevations=elevations,
                resolution=resolution
            )

            logger.info(
                f"Created terrain model with resolution {resolution}m: "
                f"{len(x_coords)}x{len(y_coords)} grid"
            )

        self.site_model.terrain = terrain_data
        return terrain_data

    def obstacle_modeling(
        self,
        obstacle_type: str,
        dimensions: Dict[str, float],
        position: Tuple[float, float, float],
        rotation: float = 0.0
    ) -> Obstacle:
        """
        Create parametric obstacle (building, tree, structure).

        Args:
            obstacle_type: Type of obstacle ('box', 'cylinder', 'tree', 'custom')
            dimensions: Dimensions dict (varies by type)
            position: (x, y, z) position of obstacle base
            rotation: Rotation around z-axis in degrees

        Returns:
            Obstacle object

        Raises:
            ValueError: If obstacle type or dimensions are invalid
        """
        if obstacle_type == "box":
            obstacle = self._create_box_obstacle(dimensions, position, rotation)
        elif obstacle_type == "cylinder":
            obstacle = self._create_cylinder_obstacle(dimensions, position)
        elif obstacle_type == "tree":
            obstacle = self._create_tree_obstacle(dimensions, position)
        else:
            raise ValueError(f"Unsupported obstacle type: {obstacle_type}")

        self.site_model.obstacles.append(obstacle)
        logger.info(f"Created {obstacle_type} obstacle at position {position}")

        return obstacle

    def horizon_profile(
        self,
        num_azimuth_points: int = 360,
        max_distance: float = 10000.0
    ) -> List[Tuple[float, float]]:
        """
        Generate horizon profile from site terrain and obstacles.

        Args:
            num_azimuth_points: Number of azimuth angles to sample
            max_distance: Maximum ray tracing distance in meters

        Returns:
            List of (azimuth, elevation) tuples in degrees
        """
        horizon_profile = []

        # Sample azimuth angles
        azimuths = np.linspace(0, 360, num_azimuth_points, endpoint=False)

        # Get observer position (center of site or specified location)
        if self.site_model.terrain:
            observer_x = np.mean(self.site_model.terrain.x_coordinates)
            observer_y = np.mean(self.site_model.terrain.y_coordinates)
            observer_z = self._get_terrain_elevation(observer_x, observer_y)
        else:
            observer_x, observer_y, observer_z = 0.0, 0.0, 0.0

        # Ray trace for each azimuth
        for azimuth in azimuths:
            elevation = self._trace_horizon_ray(
                observer_x, observer_y, observer_z,
                azimuth, max_distance
            )
            horizon_profile.append((azimuth, elevation))

        logger.info(f"Generated horizon profile with {len(horizon_profile)} points")

        return horizon_profile

    def sun_path_3d(
        self,
        sun_path_points: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Convert sun path to 3D coordinates for visualization.

        Args:
            sun_path_points: List of (azimuth, elevation, radius) tuples

        Returns:
            Numpy array of 3D coordinates (N x 3)
        """
        points_3d = []

        for azimuth, elevation, radius in sun_path_points:
            # Convert spherical to Cartesian coordinates
            x, y, z = self._spherical_to_cartesian(azimuth, elevation, radius)
            points_3d.append([x, y, z])

        return np.array(points_3d)

    def visualize_system_3d(
        self,
        array_geometry: ArrayGeometry,
        num_rows: int,
        modules_per_row: int,
        include_terrain: bool = True,
        include_obstacles: bool = True
    ) -> Dict[str, Any]:
        """
        Create 3D visualization data for PV system on terrain.

        Args:
            array_geometry: Array configuration
            num_rows: Number of rows in the array
            modules_per_row: Number of modules per row
            include_terrain: Include terrain mesh
            include_obstacles: Include obstacle meshes

        Returns:
            Dictionary containing visualization data (meshes, points, etc.)
        """
        viz_data = {
            "modules": [],
            "terrain": None,
            "obstacles": [],
            "metadata": {
                "num_rows": num_rows,
                "modules_per_row": modules_per_row,
                "total_modules": num_rows * modules_per_row
            }
        }

        # Generate module positions
        module_positions = self._generate_module_positions(
            array_geometry,
            num_rows,
            modules_per_row
        )

        viz_data["modules"] = module_positions

        # Add terrain mesh
        if include_terrain and self.site_model.terrain:
            viz_data["terrain"] = self._create_terrain_mesh()

        # Add obstacle meshes
        if include_obstacles:
            viz_data["obstacles"] = [
                self._create_obstacle_mesh(obs)
                for obs in self.site_model.obstacles
            ]

        logger.info(
            f"Generated 3D visualization with {len(module_positions)} modules, "
            f"terrain: {include_terrain}, obstacles: {len(viz_data['obstacles'])}"
        )

        return viz_data

    # Private helper methods

    def _import_obj(self, file_path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Import OBJ file format."""
        vertices = []
        faces = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    # Vertex
                    parts = line.split()[1:]
                    vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                elif line.startswith('f '):
                    # Face (convert to 0-indexed)
                    parts = line.split()[1:]
                    face = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(face)

        return vertices, faces

    def _import_stl(self, file_path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Import STL file format (ASCII or binary)."""
        vertices = []
        faces = []
        vertex_map = {}
        vertex_index = 0

        # Try ASCII first
        try:
            with open(file_path, 'r') as f:
                current_face = []
                for line in f:
                    line = line.strip()
                    if line.startswith('vertex'):
                        parts = line.split()[1:]
                        vertex = (float(parts[0]), float(parts[1]), float(parts[2]))

                        # Deduplicate vertices
                        if vertex not in vertex_map:
                            vertex_map[vertex] = vertex_index
                            vertices.append(vertex)
                            vertex_index += 1

                        current_face.append(vertex_map[vertex])

                    elif line.startswith('endfacet'):
                        if len(current_face) == 3:
                            faces.append(current_face)
                        current_face = []
        except UnicodeDecodeError:
            # Binary STL - simplified implementation
            logger.warning("Binary STL import is simplified - use ASCII for better results")
            # Return placeholder geometry
            vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            faces = [[0, 1, 2]]

        return vertices, faces

    def _import_dxf(self, file_path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Import DXF file format."""
        # Simplified DXF import - full implementation would use ezdxf library
        logger.warning("DXF import is simplified - install ezdxf for full support")

        # Return placeholder geometry
        vertices = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)]
        faces = [[0, 1, 2, 3]]

        return vertices, faces

    def _import_ply(self, file_path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Import PLY file format."""
        vertices = []
        faces = []
        reading_vertices = False
        reading_faces = False
        vertex_count = 0
        face_count = 0
        vertices_read = 0
        faces_read = 0

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                elif line.startswith('element face'):
                    face_count = int(line.split()[2])
                elif line == 'end_header':
                    reading_vertices = True
                    continue

                if reading_vertices and vertices_read < vertex_count:
                    parts = line.split()
                    vertices.append((float(parts[0]), float(parts[1]), float(parts[2])))
                    vertices_read += 1
                    if vertices_read == vertex_count:
                        reading_vertices = False
                        reading_faces = True

                elif reading_faces and faces_read < face_count:
                    parts = line.split()
                    num_vertices = int(parts[0])
                    face = [int(parts[i + 1]) for i in range(num_vertices)]
                    faces.append(face)
                    faces_read += 1

        return vertices, faces

    def _import_sketchup(self, file_path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
        """Import SketchUp file format."""
        # SketchUp import requires specialized library (sketchup-api-bridge or similar)
        logger.warning("SketchUp import requires specialized library - using placeholder")

        # Return placeholder geometry
        vertices = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0),
                    (0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)]
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

        return vertices, faces

    def _apply_transformations(
        self,
        vertices: List[Tuple[float, float, float]],
        scale: float,
        offset: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Apply scale and offset transformations to vertices."""
        transformed = []
        for v in vertices:
            transformed.append((
                v[0] * scale + offset[0],
                v[1] * scale + offset[1],
                v[2] * scale + offset[2]
            ))
        return transformed

    def _interpolate_terrain(
        self,
        survey_data: pd.DataFrame,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        method: str
    ) -> List[List[float]]:
        """Interpolate terrain elevation on regular grid."""
        from scipy.interpolate import griddata

        # Extract points and values
        points = survey_data[['x', 'y']].values
        values = survey_data['elevation'].values

        # Create grid
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)

        # Interpolate
        grid_z = griddata(points, values, (grid_x, grid_y), method=method)

        # Replace NaN with nearest neighbor
        if np.any(np.isnan(grid_z)):
            grid_z_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)

        return grid_z.tolist()

    def _create_box_obstacle(
        self,
        dimensions: Dict[str, float],
        position: Tuple[float, float, float],
        rotation: float
    ) -> Obstacle:
        """Create box-shaped obstacle."""
        width = dimensions.get('width', 10.0)
        length = dimensions.get('length', 10.0)
        height = dimensions.get('height', 5.0)

        # Create box vertices
        hw, hl, hh = width / 2, length / 2, height / 2
        vertices = [
            (-hw, -hl, 0), (hw, -hl, 0), (hw, hl, 0), (-hw, hl, 0),
            (-hw, -hl, height), (hw, -hl, height), (hw, hl, height), (-hw, hl, height)
        ]

        # Apply rotation around z-axis
        if rotation != 0:
            vertices = self._rotate_vertices(vertices, rotation)

        # Apply position offset
        vertices = [(v[0] + position[0], v[1] + position[1], v[2] + position[2]) for v in vertices]

        # Define faces
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
            [0, 1, 5, 4],  # Front
            [1, 2, 6, 5],  # Right
            [2, 3, 7, 6],  # Back
            [3, 0, 4, 7]   # Left
        ]

        return Obstacle(
            name=f"box_obstacle_{len(self.site_model.obstacles)}",
            vertices=vertices,
            faces=faces,
            height=height,
            obstacle_type="building"
        )

    def _create_cylinder_obstacle(
        self,
        dimensions: Dict[str, float],
        position: Tuple[float, float, float]
    ) -> Obstacle:
        """Create cylindrical obstacle (pole, chimney, etc.)."""
        radius = dimensions.get('radius', 1.0)
        height = dimensions.get('height', 10.0)
        segments = dimensions.get('segments', 16)

        vertices = []
        faces = []

        # Bottom circle
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle) + position[0]
            y = radius * np.sin(angle) + position[1]
            vertices.append((x, y, position[2]))

        # Top circle
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle) + position[0]
            y = radius * np.sin(angle) + position[1]
            vertices.append((x, y, position[2] + height))

        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i, next_i, next_i + segments, i + segments])

        return Obstacle(
            name=f"cylinder_obstacle_{len(self.site_model.obstacles)}",
            vertices=vertices,
            faces=faces,
            height=height,
            obstacle_type="pole"
        )

    def _create_tree_obstacle(
        self,
        dimensions: Dict[str, float],
        position: Tuple[float, float, float]
    ) -> Obstacle:
        """Create tree-shaped obstacle (simplified cone)."""
        crown_radius = dimensions.get('crown_radius', 3.0)
        height = dimensions.get('height', 10.0)
        segments = 8

        vertices = []
        faces = []

        # Base circle
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = crown_radius * np.cos(angle) + position[0]
            y = crown_radius * np.sin(angle) + position[1]
            vertices.append((x, y, position[2]))

        # Apex
        vertices.append((position[0], position[1], position[2] + height))
        apex_idx = len(vertices) - 1

        # Triangular faces
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([i, next_i, apex_idx])

        return Obstacle(
            name=f"tree_obstacle_{len(self.site_model.obstacles)}",
            vertices=vertices,
            faces=faces,
            height=height,
            obstacle_type="tree"
        )

    def _rotate_vertices(
        self,
        vertices: List[Tuple[float, float, float]],
        rotation_degrees: float
    ) -> List[Tuple[float, float, float]]:
        """Rotate vertices around z-axis."""
        rotation_rad = np.radians(rotation_degrees)
        cos_r = np.cos(rotation_rad)
        sin_r = np.sin(rotation_rad)

        rotated = []
        for x, y, z in vertices:
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            rotated.append((new_x, new_y, z))

        return rotated

    def _get_terrain_elevation(self, x: float, y: float) -> float:
        """Get terrain elevation at specific (x, y) coordinate."""
        if not self.site_model.terrain:
            return 0.0

        # Bilinear interpolation
        x_coords = np.array(self.site_model.terrain.x_coordinates)
        y_coords = np.array(self.site_model.terrain.y_coordinates)

        # Find surrounding grid points
        x_idx = np.searchsorted(x_coords, x)
        y_idx = np.searchsorted(y_coords, y)

        # Clamp to grid boundaries
        x_idx = max(1, min(x_idx, len(x_coords) - 1))
        y_idx = max(1, min(y_idx, len(y_coords) - 1))

        # Get elevation (simplified - just use nearest)
        elevation = self.site_model.terrain.elevations[y_idx][x_idx]

        return elevation

    def _trace_horizon_ray(
        self,
        observer_x: float,
        observer_y: float,
        observer_z: float,
        azimuth: float,
        max_distance: float
    ) -> float:
        """Trace ray to find horizon elevation angle."""
        max_elevation = 0.0

        # Ray direction
        azimuth_rad = np.radians(azimuth)
        dx = np.sin(azimuth_rad)
        dy = np.cos(azimuth_rad)

        # Sample along ray
        num_samples = 100
        for i in range(1, num_samples + 1):
            distance = max_distance * i / num_samples
            x = observer_x + dx * distance
            y = observer_y + dy * distance

            # Check terrain
            if self.site_model.terrain:
                terrain_z = self._get_terrain_elevation(x, y)
                elevation_angle = np.degrees(np.arctan2(terrain_z - observer_z, distance))
                max_elevation = max(max_elevation, elevation_angle)

            # Check obstacles
            for obstacle in self.site_model.obstacles:
                # Simplified - check if point is near obstacle
                # Full implementation would do proper ray-mesh intersection
                pass

        return max_elevation

    def _spherical_to_cartesian(
        self,
        azimuth: float,
        elevation: float,
        radius: float
    ) -> Tuple[float, float, float]:
        """Convert spherical coordinates to Cartesian."""
        azimuth_rad = np.radians(azimuth)
        elevation_rad = np.radians(elevation)

        x = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
        z = radius * np.sin(elevation_rad)

        return x, y, z

    def _generate_module_positions(
        self,
        array_geometry: ArrayGeometry,
        num_rows: int,
        modules_per_row: int
    ) -> List[Dict[str, Any]]:
        """Generate 3D positions for all modules."""
        module_positions = []

        # Calculate row pitch
        pitch = array_geometry.row_spacing

        for row in range(num_rows):
            for col in range(modules_per_row):
                # Calculate position
                x = col * array_geometry.module_width
                y = row * pitch
                z = self._get_terrain_elevation(x, y) if self.site_model.terrain else 0.0

                # Calculate module corners
                corners = self._calculate_module_corners(
                    x, y, z,
                    array_geometry.module_width,
                    array_geometry.module_height,
                    array_geometry.tilt,
                    array_geometry.azimuth
                )

                module_positions.append({
                    "row": row,
                    "column": col,
                    "position": (x, y, z),
                    "corners": corners,
                    "tilt": array_geometry.tilt,
                    "azimuth": array_geometry.azimuth
                })

        return module_positions

    def _calculate_module_corners(
        self,
        x: float,
        y: float,
        z: float,
        width: float,
        height: float,
        tilt: float,
        azimuth: float
    ) -> List[Tuple[float, float, float]]:
        """Calculate 3D coordinates of module corners."""
        # Start with flat rectangle
        hw, hh = width / 2, height / 2
        corners = [
            (-hw, -hh, 0),
            (hw, -hh, 0),
            (hw, hh, 0),
            (-hw, hh, 0)
        ]

        # Apply tilt rotation (around x-axis)
        tilt_rad = np.radians(tilt)
        corners = [
            (c[0], c[1] * np.cos(tilt_rad), c[1] * np.sin(tilt_rad))
            for c in corners
        ]

        # Apply azimuth rotation (around z-axis)
        azimuth_rad = np.radians(azimuth)
        cos_az = np.cos(azimuth_rad)
        sin_az = np.sin(azimuth_rad)
        corners = [
            (c[0] * cos_az - c[1] * sin_az, c[0] * sin_az + c[1] * cos_az, c[2])
            for c in corners
        ]

        # Translate to position
        corners = [(c[0] + x, c[1] + y, c[2] + z) for c in corners]

        return corners

    def _create_terrain_mesh(self) -> Dict[str, Any]:
        """Create mesh data for terrain visualization."""
        if not self.site_model.terrain:
            return {}

        return {
            "x": self.site_model.terrain.x_coordinates,
            "y": self.site_model.terrain.y_coordinates,
            "z": self.site_model.terrain.elevations,
            "type": "surface"
        }

    def _create_obstacle_mesh(self, obstacle: Obstacle) -> Dict[str, Any]:
        """Create mesh data for obstacle visualization."""
        return {
            "vertices": obstacle.vertices,
            "faces": obstacle.faces,
            "name": obstacle.name,
            "type": obstacle.obstacle_type
        }
