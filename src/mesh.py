"""
Mesh Generation Module
======================

Provides uniform triangular meshes for:
1. FEM discretization (forward problem)
2. Source candidate locations (distributed inverse problem)

Requires gmsh for quality mesh generation.
"""

import numpy as np
from typing import Tuple, Optional, List

# Check gmsh availability at import time
_GMSH_AVAILABLE = False
_GMSH_ERROR = None

try:
    import gmsh
    _GMSH_AVAILABLE = True
except ImportError as e:
    _GMSH_ERROR = f"gmsh not installed: {e}"
except OSError as e:
    _GMSH_ERROR = f"gmsh library dependencies missing: {e}"


def _require_gmsh():
    """Raise error if gmsh is not available."""
    if not _GMSH_AVAILABLE:
        raise RuntimeError(
            f"gmsh is required for mesh generation but is not available.\n"
            f"Error: {_GMSH_ERROR}\n"
            f"Install with: pip install gmsh\n"
            f"On Linux, you may also need: apt-get install libglu1-mesa libxft2"
        )


# =============================================================================
# SENSOR LOCATION UTILITIES
# =============================================================================

def get_disk_sensor_locations(n_sensors: int = 100, radius: float = 1.0) -> np.ndarray:
    """
    Get evenly spaced sensor locations on disk boundary.
    
    Parameters
    ----------
    n_sensors : int
        Number of sensors
    radius : float
        Disk radius
        
    Returns
    -------
    locations : array, shape (n_sensors, 2)
        Sensor (x, y) coordinates
    """
    theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def get_ellipse_sensor_locations(a: float, b: float, n_sensors: int = 100) -> np.ndarray:
    """
    Get evenly spaced (by parameter angle) sensor locations on ellipse boundary.
    
    Parameters
    ----------
    a, b : float
        Semi-axes of ellipse
    n_sensors : int
        Number of sensors
        
    Returns
    -------
    locations : array, shape (n_sensors, 2)
        Sensor (x, y) coordinates
    """
    theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    return np.column_stack([a * np.cos(theta), b * np.sin(theta)])


def get_polygon_sensor_locations(vertices: List[Tuple[float, float]], 
                                  n_sensors: int = 100) -> np.ndarray:
    """
    Get evenly spaced (by arc length) sensor locations on polygon boundary.
    
    Parameters
    ----------
    vertices : list of (x, y)
        Polygon vertices in order
    n_sensors : int
        Number of sensors
        
    Returns
    -------
    locations : array, shape (n_sensors, 2)
        Sensor (x, y) coordinates
    """
    vertices = np.array(vertices)
    n_verts = len(vertices)
    
    # Compute edge lengths and cumulative arc length
    edges = []
    edge_lengths = []
    for i in range(n_verts):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_verts]
        edges.append((p1, p2))
        edge_lengths.append(np.linalg.norm(p2 - p1))
    
    total_length = sum(edge_lengths)
    cumulative = np.cumsum([0] + edge_lengths)
    
    # Place sensors evenly by arc length
    sensor_positions = np.linspace(0, total_length, n_sensors, endpoint=False)
    locations = []
    
    for s in sensor_positions:
        # Find which edge this position is on
        edge_idx = np.searchsorted(cumulative[1:], s, side='right')
        edge_idx = min(edge_idx, n_verts - 1)
        
        # Interpolate along edge
        s_on_edge = s - cumulative[edge_idx]
        t = s_on_edge / edge_lengths[edge_idx] if edge_lengths[edge_idx] > 0 else 0
        t = np.clip(t, 0, 1)
        
        p1, p2 = edges[edge_idx]
        pt = p1 + t * (p2 - p1)
        locations.append(pt)
    
    return np.array(locations)


def create_disk_mesh(resolution: float = 0.1, radius: float = 1.0, 
                     interior_only: bool = False,
                     sensor_locations: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform triangular mesh of the disk using gmsh.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size (smaller = finer mesh)
    radius : float
        Disk radius
    sensor_locations : array, shape (n_sensors, 2), optional
        Fixed boundary points that must be included as mesh nodes.
        These represent physical sensor/measurement locations.
        
    Returns
    -------
    nodes : array, shape (N, 2)
        Node coordinates
    elements : array, shape (M, 3)
        Triangle connectivity (node indices)
    boundary_indices : array
        Indices of boundary nodes
    interior_indices : array
        Indices of interior nodes
    sensor_indices : array
        Indices of sensor locations in nodes array (empty if no sensors)
        
    Raises
    ------
    RuntimeError
        If gmsh is not available
    """
    _require_gmsh()
    return _create_mesh_gmsh(resolution, radius, sensor_locations)


def _create_mesh_gmsh(resolution: float, radius: float,
                      sensor_locations: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh using gmsh with sensor points exactly embedded as boundary nodes.
    
    Strategy: Use GEO kernel with circular arcs between sensor points. This ensures
    sensors are exactly on mesh nodes without requiring gmsh.model.mesh.embed().
    Uses MathEval background field for uniform mesh resolution control.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size
    radius : float
        Disk radius
    sensor_locations : array, shape (n_sensors, 2), optional
        Sensor locations on boundary. Will be exactly embedded as mesh nodes.
        
    Returns
    -------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    boundary_indices : array
    interior_indices : array
    sensor_indices : array
        Indices into nodes array for sensors (exactly embedded)
    """
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("disk")
    
    if sensor_locations is not None and len(sensor_locations) > 0:
        # Use GEO kernel with circular arcs between sensor points
        n_sensors = len(sensor_locations)
        center = gmsh.model.geo.addPoint(0, 0, 0)
        boundary_pts = []
        for x, y in sensor_locations:
            pt = gmsh.model.geo.addPoint(x, y, 0)
            boundary_pts.append(pt)
        arcs = []
        for i in range(n_sensors):
            arc = gmsh.model.geo.addCircleArc(boundary_pts[i], center, boundary_pts[(i + 1) % n_sensors])
            arcs.append(arc)
        loop = gmsh.model.geo.addCurveLoop(arcs)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
    else:
        # No sensors - use OCC kernel for simple disk
        disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
        gmsh.model.occ.synchronize()
    
    # Use MathEval background field for uniform resolution
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    field_id = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_id, "F", str(resolution))
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    
    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    nodes = np.zeros((n_nodes, 2))
    for i, tag in enumerate(node_tags):
        idx = tag_to_idx[int(tag)]
        nodes[idx, 0] = node_coords[3*i]
        nodes[idx, 1] = node_coords[3*i + 1]
    
    # Extract triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    elements = []
    for i, elem_type in enumerate(elem_types):
        if elem_type == 2:  # 3-node triangle
            node_tags_flat = elem_node_tags[i]
            n_elems = len(node_tags_flat) // 3
            for j in range(n_elems):
                elements.append([tag_to_idx[int(node_tags_flat[3*j])],
                                 tag_to_idx[int(node_tags_flat[3*j + 1])],
                                 tag_to_idx[int(node_tags_flat[3*j + 2])]])
    elements = np.array(elements)
    gmsh.finalize()
    
    # Classify nodes by radius
    radii = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    boundary_indices = np.where(radii > 0.99 * radius)[0]
    interior_indices = np.where(radii <= 0.99 * radius)[0]
    
    # Find sensor indices - they should now be exactly embedded
    sensor_indices = np.array([], dtype=int)
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((nodes[:, 0] - sx)**2 + (nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
            if dists[sensor_indices[i]] > 1e-10:
                raise RuntimeError(f"Sensor {i} not embedded correctly")
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def _create_mesh_fallback(resolution: float, radius: float,
                          sensor_locations: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback mesh using sunflower pattern + Delaunay.
    Produces quasi-uniform distribution without gmsh.
    
    If sensor_locations provided, uses those as boundary points.
    Otherwise generates evenly spaced boundary points.
    """
    from scipy.spatial import Delaunay
    
    # Estimate number of points
    area = np.pi * radius**2
    n_total = int(area / (resolution**2 * 0.5))  # rough estimate
    
    # Boundary points - use sensors if provided, otherwise uniform
    if sensor_locations is not None and len(sensor_locations) > 0:
        boundary_pts = np.array(sensor_locations)
        n_boundary = len(boundary_pts)
    else:
        n_boundary = int(2 * np.pi * radius / resolution)
        theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        boundary_pts = np.column_stack([
            radius * np.cos(theta_boundary),
            radius * np.sin(theta_boundary)
        ])
    
    n_interior = n_total - n_boundary
    
    # Interior points using sunflower/Vogel pattern (uniform area distribution)
    indices = np.arange(1, n_interior + 1)
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees
    
    r_int = radius * np.sqrt(indices / n_interior)  # sqrt for uniform area (no gap)
    theta_int = golden_angle * indices
    
    interior_pts = np.column_stack([
        r_int * np.cos(theta_int),
        r_int * np.sin(theta_int)
    ])
    
    # Combine: boundary first, then interior
    nodes = np.vstack([boundary_pts, interior_pts])
    
    # Triangulate
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # Remove triangles outside disk
    centroids = nodes[elements].mean(axis=1)
    inside = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2) < radius * 1.01
    elements = elements[inside]
    
    # Indices
    boundary_indices = np.arange(n_boundary)
    interior_indices = np.arange(n_boundary, len(nodes))
    
    # Sensor indices are just the first n_sensor boundary indices (they ARE the boundary)
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.arange(len(sensor_locations))
    else:
        sensor_indices = np.array([], dtype=int)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def get_source_grid(resolution: float = 0.15, radius: float = 1.0) -> np.ndarray:
    """
    Get interior points for source candidate locations (unit disk).
    
    Uses gmsh to generate a quality mesh, then extracts interior nodes.
    This ensures uniform point distribution matching FEM mesh quality.
    
    Parameters
    ----------
    resolution : float
        Grid resolution (mesh element size)
    radius : float
        Maximum radius for source locations. Default 1.0 uses all interior nodes.
        Set < 1.0 to exclude sources near the boundary.
        
    Returns
    -------
    points : array, shape (M, 2)
        Interior mesh nodes as source candidate locations
    """
    result = create_disk_mesh(resolution, radius=1.0)
    nodes, elements, boundary_idx, interior_idx = result[0], result[1], result[2], result[3]
    
    interior_points = nodes[interior_idx]
    
    # Only filter by radius if explicitly requested (radius < 1.0)
    if radius < 1.0:
        radii = np.sqrt(interior_points[:, 0]**2 + interior_points[:, 1]**2)
        mask = radii < radius
        return interior_points[mask]
    
    return interior_points


def plot_mesh(nodes: np.ndarray, elements: np.ndarray, 
              boundary_indices: np.ndarray = None,
              interior_indices: np.ndarray = None,
              title: str = "Mesh"):
    """Visualize mesh for debugging."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot triangles
    triangles = nodes[elements]
    pc = PolyCollection(triangles, facecolors='lightblue', 
                        edgecolors='gray', linewidths=0.5, alpha=0.7)
    ax.add_collection(pc)
    
    # Plot nodes
    if interior_indices is not None:
        ax.plot(nodes[interior_indices, 0], nodes[interior_indices, 1], 
                'b.', markersize=3, label=f'Interior ({len(interior_indices)})')
    if boundary_indices is not None:
        ax.plot(nodes[boundary_indices, 0], nodes[boundary_indices, 1], 
                'r.', markersize=5, label=f'Boundary ({len(boundary_indices)})')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n{len(nodes)} nodes, {len(elements)} elements")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


# =============================================================================
# ELLIPSE MESH GENERATION
# =============================================================================

def create_ellipse_mesh(a: float = 2.0, b: float = 1.0, resolution: float = 0.1,
                        sensor_locations: np.ndarray = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform triangular mesh of an ellipse.
    
    Parameters
    ----------
    a : float
        Semi-major axis (along x)
    b : float
        Semi-minor axis (along y)
    resolution : float
        Target mesh element size
    sensor_locations : array, shape (n_sensors, 2), optional
        Fixed boundary points that must be mesh nodes
        
    Returns
    -------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    boundary_indices : array
    interior_indices : array
    sensor_indices : array
        Indices of sensor locations (empty if no sensors)
        
    Raises
    ------
    RuntimeError
        If gmsh is not available
    """
    _require_gmsh()
    return _create_ellipse_mesh_gmsh(a, b, resolution, sensor_locations)


def _create_ellipse_mesh_gmsh(a: float, b: float, resolution: float,
                              sensor_locations: np.ndarray = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create ellipse mesh using gmsh with sensor points exactly embedded.
    
    Strategy: Use GEO kernel with splines between sensor points (like disk uses arcs).
    This ensures sensors are exactly on mesh nodes.
    """
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ellipse")
    
    if sensor_locations is not None and len(sensor_locations) > 0:
        # Use GEO kernel with splines between sensor points
        n_sensors = len(sensor_locations)
        
        # Create boundary points
        boundary_pts = []
        for x, y in sensor_locations:
            pt = gmsh.model.geo.addPoint(x, y, 0)
            boundary_pts.append(pt)
        
        # Create line segments between consecutive sensor points
        # For ellipse, use lines (which gmsh will refine) rather than arcs
        lines = []
        for i in range(n_sensors):
            line = gmsh.model.geo.addLine(boundary_pts[i], boundary_pts[(i + 1) % n_sensors])
            lines.append(line)
        
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        
    else:
        # No sensors - use OCC kernel for simple ellipse
        ellipse = gmsh.model.occ.addDisk(0, 0, 0, a, b)
        gmsh.model.occ.synchronize()
    
    # Use MathEval background field for uniform resolution
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    field_id = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_id, "F", str(resolution))
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    gmsh.model.mesh.generate(2)
    
    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    
    nodes = np.zeros((n_nodes, 2))
    for i, tag in enumerate(node_tags):
        idx = tag_to_idx[int(tag)]
        nodes[idx, 0] = node_coords[3*i]
        nodes[idx, 1] = node_coords[3*i + 1]
    
    # Extract triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    elements = []
    for i, elem_type in enumerate(elem_types):
        if elem_type == 2:
            node_tags_flat = elem_node_tags[i]
            n_elems = len(node_tags_flat) // 3
            for j in range(n_elems):
                n1 = tag_to_idx[int(node_tags_flat[3*j])]
                n2 = tag_to_idx[int(node_tags_flat[3*j + 1])]
                n3 = tag_to_idx[int(node_tags_flat[3*j + 2])]
                elements.append([n1, n2, n3])
    elements = np.array(elements)
    
    gmsh.finalize()
    
    # Classify nodes (on ellipse boundary if (x/a)² + (y/b)² ≈ 1)
    ellipse_param = (nodes[:, 0]/a)**2 + (nodes[:, 1]/b)**2
    boundary_indices = np.where(ellipse_param > 0.98)[0]
    interior_indices = np.where(ellipse_param <= 0.98)[0]
    
    # Find sensor indices - they should now be exactly embedded
    sensor_indices = np.array([], dtype=int)
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((nodes[:, 0] - sx)**2 + (nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def _create_ellipse_mesh_fallback(a: float, b: float, resolution: float,
                                  sensor_locations: np.ndarray = None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback ellipse mesh using sunflower pattern + Delaunay."""
    from scipy.spatial import Delaunay
    
    # Estimate number of points
    area = np.pi * a * b
    n_total = int(area / (resolution**2 * 0.5))
    perimeter = np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))  # Ramanujan approx
    
    # Boundary points - use sensors if provided
    if sensor_locations is not None and len(sensor_locations) > 0:
        boundary_pts = np.array(sensor_locations)
        n_boundary = len(boundary_pts)
    else:
        n_boundary = max(int(perimeter / resolution), 20)
        theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        boundary_pts = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    
    n_interior = n_total - n_boundary
    
    # Interior points using transformed sunflower pattern
    indices = np.arange(1, n_interior + 1)
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # Generate in unit disk, then scale to ellipse
    r = np.sqrt(indices / n_interior)  # no gap - use full interior
    phi = golden_angle * indices
    interior_pts = np.column_stack([a * r * np.cos(phi), b * r * np.sin(phi)])
    
    nodes = np.vstack([boundary_pts, interior_pts])
    
    # Triangulate
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # Remove triangles outside ellipse
    centroids = nodes[elements].mean(axis=1)
    inside = (centroids[:, 0]/a)**2 + (centroids[:, 1]/b)**2 < 1.01
    elements = elements[inside]
    
    boundary_indices = np.arange(n_boundary)
    interior_indices = np.arange(n_boundary, len(nodes))
    
    # Sensor indices
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.arange(len(sensor_locations))
    else:
        sensor_indices = np.array([], dtype=int)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def get_ellipse_source_grid(a: float = 2.0, b: float = 1.0, 
                            resolution: float = 0.15, margin: float = 0.0
                            ) -> np.ndarray:
    """
    Get interior points for source candidates in an ellipse.
    
    Uses gmsh to generate a quality mesh, then extracts interior nodes.
    This ensures uniform point distribution matching FEM mesh quality.
    
    Parameters
    ----------
    a, b : float
        Semi-axes of ellipse
    resolution : float
        Grid resolution (mesh element size)
    margin : float
        Optional: exclude sources within this fraction of boundary.
        Default 0.0 means use all interior mesh nodes.
        
    Returns
    -------
    points : array, shape (M, 2)
        Interior mesh nodes as source candidate locations
    """
    result = create_ellipse_mesh(a, b, resolution)
    nodes, _, _, interior_idx = result[0], result[1], result[2], result[3]
    interior_points = nodes[interior_idx]
    
    # Only filter by margin if explicitly requested (margin > 0)
    if margin > 0:
        ellipse_param = (interior_points[:, 0]/a)**2 + (interior_points[:, 1]/b)**2
        mask = ellipse_param < (1 - margin)**2
        return interior_points[mask]
    
    return interior_points


# =============================================================================
# POLYGON MESH GENERATION
# =============================================================================

def create_polygon_mesh(vertices: List[Tuple[float, float]], resolution: float = 0.1,
                        sensor_locations: np.ndarray = None
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform triangular mesh of a polygon.
    
    Parameters
    ----------
    vertices : list of (x, y) tuples
        Polygon vertices in order (CCW for standard orientation)
    resolution : float
        Target mesh element size
    sensor_locations : array, shape (n_sensors, 2), optional
        Fixed boundary points that must be mesh nodes
        
    Returns
    -------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    boundary_indices : array
    interior_indices : array
    sensor_indices : array
    
    Raises
    ------
    RuntimeError
        If gmsh is not available
    
    Examples
    --------
    >>> # Unit square
    >>> verts = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    >>> nodes, elems, b_idx, i_idx, s_idx = create_polygon_mesh(verts, resolution=0.2)
    
    >>> # L-shaped domain
    >>> verts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    >>> nodes, elems, b_idx, i_idx, s_idx = create_polygon_mesh(verts, resolution=0.15)
    """
    _require_gmsh()
    return _create_polygon_mesh_gmsh(vertices, resolution, sensor_locations)


def _create_polygon_mesh_gmsh(vertices: List[Tuple[float, float]], resolution: float,
                              sensor_locations: np.ndarray = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create polygon mesh using gmsh with sensor points exactly embedded.
    
    Strategy: Use GEO kernel with all boundary points (vertices + sensors) as explicit
    nodes connected by lines. This ensures sensors are exactly on mesh nodes.
    """
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polygon")
    
    vertices_arr = np.array(vertices)
    n_verts = len(vertices)
    
    if sensor_locations is not None and len(sensor_locations) > 0:
        # Merge vertices and sensors, ordered around the boundary
        all_boundary_pts = _merge_boundary_points(vertices_arr, sensor_locations)
        n_pts = len(all_boundary_pts)
        
        # Create points using GEO kernel
        point_tags = []
        for x, y in all_boundary_pts:
            pt = gmsh.model.geo.addPoint(x, y, 0)
            point_tags.append(pt)
        
        # Create lines between consecutive points
        lines = []
        for i in range(n_pts):
            line = gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n_pts])
            lines.append(line)
        
        loop = gmsh.model.geo.addCurveLoop(lines)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
    else:
        # No sensors - use OCC kernel for simple polygon
        vertex_points = []
        for x, y in vertices:
            vertex_points.append(gmsh.model.occ.addPoint(x, y, 0, resolution))
        
        lines = []
        for i in range(n_verts):
            lines.append(gmsh.model.occ.addLine(vertex_points[i], vertex_points[(i+1) % n_verts]))
        
        curve_loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()
    
    # Use MathEval background field for uniform resolution
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    field_id = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(field_id, "F", str(resolution))
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    
    gmsh.model.mesh.generate(2)
    
    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    
    nodes = np.zeros((n_nodes, 2))
    for i, tag in enumerate(node_tags):
        idx = tag_to_idx[int(tag)]
        nodes[idx, 0] = node_coords[3*i]
        nodes[idx, 1] = node_coords[3*i + 1]
    
    # Extract triangles
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    elements = []
    for i, elem_type in enumerate(elem_types):
        if elem_type == 2:
            node_tags_flat = elem_node_tags[i]
            n_elems = len(node_tags_flat) // 3
            for j in range(n_elems):
                n1 = tag_to_idx[int(node_tags_flat[3*j])]
                n2 = tag_to_idx[int(node_tags_flat[3*j + 1])]
                n3 = tag_to_idx[int(node_tags_flat[3*j + 2])]
                elements.append([n1, n2, n3])
    elements = np.array(elements)
    
    gmsh.finalize()
    
    # Classify boundary nodes (on polygon edges)
    boundary_indices = _find_polygon_boundary_nodes(nodes, vertices_arr, tol=resolution*0.5)
    interior_indices = np.setdiff1d(np.arange(len(nodes)), boundary_indices)
    
    # Find sensor indices - they should now be exactly embedded
    sensor_indices = np.array([], dtype=int)
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((nodes[:, 0] - sx)**2 + (nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def _merge_boundary_points(vertices: np.ndarray, sensors: np.ndarray) -> np.ndarray:
    """Merge vertices and sensor points, ordered around the polygon boundary.
    
    For each edge (v_i, v_{i+1}), we find sensors on that edge and sort them
    by parametric position along the edge.
    """
    n_verts = len(vertices)
    all_points = []
    
    for i in range(n_verts):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n_verts]
        
        # Add the vertex
        all_points.append(v1)
        
        # Find sensors on this edge
        edge_vec = v2 - v1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            continue
        edge_dir = edge_vec / edge_len
        edge_normal = np.array([-edge_dir[1], edge_dir[0]])
        
        sensors_on_edge = []
        for j, s in enumerate(sensors):
            # Check if sensor is on this edge
            to_sensor = s - v1
            along = np.dot(to_sensor, edge_dir)
            perp = np.abs(np.dot(to_sensor, edge_normal))
            
            # On edge if: 0 < along < edge_len and perp ≈ 0
            if 0.01 * edge_len < along < 0.99 * edge_len and perp < 1e-6:
                sensors_on_edge.append((along, s))
        
        # Sort sensors by position along edge
        sensors_on_edge.sort(key=lambda x: x[0])
        for _, s in sensors_on_edge:
            all_points.append(s)
    
    return np.array(all_points)
    
    # Find sensor indices
    sensor_indices = np.array([], dtype=int)
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((nodes[:, 0] - sx)**2 + (nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def _find_polygon_boundary_nodes(nodes: np.ndarray, vertices: np.ndarray, 
                                 tol: float = 0.01) -> np.ndarray:
    """Find nodes that lie on polygon boundary edges."""
    n_verts = len(vertices)
    boundary_mask = np.zeros(len(nodes), dtype=bool)
    
    for i in range(n_verts):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_verts]
        
        # Distance from each node to this edge
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            continue
        edge_unit = edge_vec / edge_len
        
        for j, node in enumerate(nodes):
            # Project node onto edge line
            v = node - p1
            t = np.dot(v, edge_unit)
            
            if 0 <= t <= edge_len:
                # Distance to edge
                proj = p1 + t * edge_unit
                dist = np.linalg.norm(node - proj)
                if dist < tol:
                    boundary_mask[j] = True
    
    return np.where(boundary_mask)[0]


def _create_polygon_mesh_fallback(vertices: List[Tuple[float, float]], resolution: float,
                                  sensor_locations: np.ndarray = None
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback polygon mesh using boundary sampling + Delaunay."""
    from scipy.spatial import Delaunay
    
    vertices_arr = np.array(vertices)
    n_verts = len(vertices)
    
    # Use sensor locations as boundary points if provided, otherwise sample edges
    if sensor_locations is not None and len(sensor_locations) > 0:
        boundary_pts = np.array(sensor_locations)
    else:
        # Sample boundary edges
        boundary_pts_list = []
        for i in range(n_verts):
            p1 = vertices_arr[i]
            p2 = vertices_arr[(i + 1) % n_verts]
            edge_len = np.linalg.norm(p2 - p1)
            n_edge_pts = max(int(edge_len / resolution), 2)
            
            for j in range(n_edge_pts):
                t = j / n_edge_pts
                boundary_pts_list.append(p1 + t * (p2 - p1))
        
        boundary_pts = np.array(boundary_pts_list)
    
    n_boundary = len(boundary_pts)
    
    # Compute bounding box
    x_min, y_min = vertices_arr.min(axis=0)
    x_max, y_max = vertices_arr.max(axis=0)
    
    # Generate interior points on a grid, keep only those inside polygon
    x_grid = np.arange(x_min + resolution/2, x_max, resolution)
    y_grid = np.arange(y_min + resolution/2, y_max, resolution)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Point-in-polygon test
    inside_mask = _points_in_polygon(grid_pts, vertices_arr)
    interior_pts = grid_pts[inside_mask]
    
    # Remove interior points that exactly coincide with boundary points
    if len(interior_pts) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(boundary_pts)
        dists, _ = tree.query(interior_pts)
        interior_pts = interior_pts[dists > resolution * 0.05]
    
    nodes = np.vstack([boundary_pts, interior_pts])
    
    # Triangulate
    if len(nodes) < 3:
        sensor_indices = np.arange(n_boundary) if sensor_locations is not None else np.array([], dtype=int)
        return nodes, np.array([]), np.arange(n_boundary), np.array([]), sensor_indices
    
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # Remove triangles outside polygon
    centroids = nodes[elements].mean(axis=1)
    inside = _points_in_polygon(centroids, vertices_arr)
    elements = elements[inside]
    
    boundary_indices = np.arange(n_boundary)
    interior_indices = np.arange(n_boundary, len(nodes))
    
    # Sensor indices - when sensors are used as boundary, they are the first n_sensor points
    if sensor_locations is not None and len(sensor_locations) > 0:
        sensor_indices = np.arange(len(sensor_locations))
    else:
        sensor_indices = np.array([], dtype=int)
    
    return nodes, elements, boundary_indices, interior_indices, sensor_indices


def _points_in_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Ray casting algorithm for point-in-polygon test."""
    n = len(vertices)
    inside = np.zeros(len(points), dtype=bool)
    
    for k, (px, py) in enumerate(points):
        count = 0
        for i in range(n):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % n]
            
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1):
                count += 1
        
        inside[k] = (count % 2 == 1)
    
    return inside


def get_polygon_source_grid(vertices: List[Tuple[float, float]], 
                            resolution: float = 0.15, margin: float = 0.0
                            ) -> np.ndarray:
    """
    Get interior points for source candidates in a polygon.
    
    Uses gmsh to generate a quality mesh, then extracts interior nodes.
    This ensures uniform point distribution matching FEM mesh quality.
    
    Parameters
    ----------
    vertices : list of (x, y)
        Polygon vertices
    resolution : float
        Grid resolution (mesh element size)
    margin : float
        Optional: exclude sources within this distance of boundary.
        Default 0.0 means use all interior mesh nodes.
        
    Returns
    -------
    points : array, shape (M, 2)
        Interior mesh nodes as source candidate locations
    """
    result = create_polygon_mesh(vertices, resolution)
    nodes, _, boundary_idx, interior_idx = result[0], result[1], result[2], result[3]
    interior_points = nodes[interior_idx]
    
    if len(interior_points) == 0:
        return interior_points
    
    # Only filter by margin if explicitly requested (margin > 0)
    if margin > 0:
        boundary_pts = nodes[boundary_idx]
        if len(boundary_pts) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(boundary_pts)
            dists, _ = tree.query(interior_points)
            mask = dists > margin
            return interior_points[mask]
    
    return interior_points


def get_brain_boundary(n_points: int = 100) -> np.ndarray:
    """
    Generate a brain-like 2D boundary (coronal cross-section).
    
    This creates a smooth closed curve resembling a 2D brain slice,
    suitable for EEG source localization demonstrations.
    
    Parameters
    ----------
    n_points : int
        Number of boundary points
        
    Returns
    -------
    boundary : array (n_points, 2)
        Boundary coordinates
    """
    t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Base ellipse (slightly wider than tall)
    a, b = 1.0, 0.8
    
    # Add shape variations:
    # - cos(2t): bilateral symmetry bulges
    # - cos(4t): slight indentations
    # - cos(3t): asymmetry hint
    r = 1.0 + 0.15 * np.cos(2*t) - 0.1 * np.cos(4*t) + 0.05 * np.cos(3*t)
    
    # Flatten bottom slightly (base of brain)
    r = r * (1 - 0.1 * np.sin(t)**4)
    
    x = a * r * np.cos(t)
    y = b * r * np.sin(t)
    
    # Center the shape
    y = y + 0.05
    
    return np.column_stack([x, y])


def create_brain_mesh(resolution: float = 0.1, 
                      sensor_locations: np.ndarray = None,
                      n_sensors: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a triangular mesh of a brain-like 2D domain.
    
    Uses polygon mesh generation with brain boundary points.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size
    sensor_locations : array, optional
        Fixed sensor locations on boundary. If None, uses n_sensors evenly spaced.
    n_sensors : int
        Number of evenly spaced sensors if sensor_locations not provided
        
    Returns
    -------
    nodes : array (N, 2)
        Node coordinates
    elements : array (M, 3)
        Triangle connectivity
    boundary_indices : array
        Indices of boundary nodes
    interior_indices : array
        Indices of interior nodes
    sensor_indices : array
        Indices of sensor locations
    """
    # Get brain boundary as polygon vertices
    boundary = get_brain_boundary(n_points=100)
    vertices = [tuple(p) for p in boundary]
    
    # If no sensors specified, use evenly spaced points on brain boundary
    if sensor_locations is None:
        sensor_locations = get_brain_boundary(n_points=n_sensors)
    
    # Use polygon mesh generator
    return create_polygon_mesh(vertices, resolution, sensor_locations)


def get_brain_source_grid(resolution: float = 0.15, margin: float = 0.0) -> np.ndarray:
    """
    Get source candidate grid for brain domain.
    
    Uses gmsh to generate a quality mesh, then extracts interior nodes.
    
    Parameters
    ----------
    resolution : float
        Grid spacing (mesh element size)
    margin : float
        Optional: distance from boundary to exclude.
        Default 0.0 uses all interior mesh nodes.
        
    Returns
    -------
    points : array (N, 2)
        Interior mesh nodes as source candidate locations
    """
    boundary = get_brain_boundary(n_points=200)
    return get_polygon_source_grid([tuple(p) for p in boundary], resolution, margin)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Mesh Generation Demo")
    print("=" * 50)
    
    # Create meshes at different resolutions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, res in zip(axes, [0.2, 0.1, 0.05]):
        nodes, elements, b_idx, i_idx = create_disk_mesh(resolution=res)
        print(f"Resolution {res}: {len(nodes)} nodes, {len(elements)} elements")
        
        # Plot
        from matplotlib.collections import PolyCollection
        triangles = nodes[elements]
        pc = PolyCollection(triangles, facecolors='lightblue',
                           edgecolors='gray', linewidths=0.3, alpha=0.7)
        ax.add_collection(pc)
        ax.plot(nodes[i_idx, 0], nodes[i_idx, 1], 'b.', markersize=2)
        ax.plot(nodes[b_idx, 0], nodes[b_idx, 1], 'r.', markersize=4)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f"res={res}\n{len(nodes)} nodes")
    
    plt.tight_layout()
    plt.savefig('mesh_demo.png', dpi=150)
    plt.show()


# =============================================================================
# MESH SAVING FUNCTIONS
# =============================================================================

def save_mesh_npz(filepath: str, nodes: np.ndarray, elements: np.ndarray,
                  boundary_indices: np.ndarray, interior_indices: np.ndarray,
                  metadata: dict = None):
    """
    Save mesh to numpy .npz format.
    
    Parameters
    ----------
    filepath : str
        Output path (will add .npz if not present)
    nodes : np.ndarray
        Node coordinates (n_nodes, 2)
    elements : np.ndarray
        Triangle connectivity (n_elements, 3)
    boundary_indices : np.ndarray
        Indices of boundary nodes
    interior_indices : np.ndarray
        Indices of interior nodes
    metadata : dict, optional
        Additional metadata (resolution, domain_type, etc.)
    """
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    
    save_dict = {
        'nodes': nodes,
        'elements': elements,
        'boundary_indices': boundary_indices,
        'interior_indices': interior_indices,
    }
    
    # Store metadata as a JSON string in the npz
    if metadata:
        import json
        save_dict['metadata'] = np.array([json.dumps(metadata)])
    
    np.savez(filepath, **save_dict)
    return filepath


def load_mesh_npz(filepath: str) -> dict:
    """
    Load mesh from numpy .npz format.
    
    Returns
    -------
    dict with keys: nodes, elements, boundary_indices, interior_indices, metadata
    """
    data = np.load(filepath, allow_pickle=True)
    result = {
        'nodes': data['nodes'],
        'elements': data['elements'],
        'boundary_indices': data['boundary_indices'],
        'interior_indices': data['interior_indices'],
    }
    
    if 'metadata' in data:
        import json
        result['metadata'] = json.loads(str(data['metadata'][0]))
    else:
        result['metadata'] = {}
    
    return result


def save_mesh_msh(filepath: str, nodes: np.ndarray, elements: np.ndarray,
                  boundary_indices: np.ndarray = None, domain_type: str = 'generic'):
    """
    Save mesh to gmsh .msh format (version 2.2 ASCII for compatibility).
    
    Parameters
    ----------
    filepath : str
        Output path (will add .msh if not present)
    nodes : np.ndarray
        Node coordinates (n_nodes, 2)
    elements : np.ndarray
        Triangle connectivity (n_elements, 3)
    boundary_indices : np.ndarray, optional
        Indices of boundary nodes (for physical group tagging)
    domain_type : str
        Name for the domain physical group
    """
    if not filepath.endswith('.msh'):
        filepath = filepath + '.msh'
    
    n_nodes = len(nodes)
    n_elements = len(elements)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")  # Version 2.2, ASCII, 8-byte floats
        f.write("$EndMeshFormat\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        for i, node in enumerate(nodes):
            f.write(f"{i+1} {node[0]:.15e} {node[1]:.15e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements
        # In gmsh format: elm-number elm-type number-of-tags <tags> node-list
        # elm-type 2 = 3-node triangle
        # We'll add boundary edges as type 1 (2-node line) if boundary_indices provided
        
        boundary_edges = []
        if boundary_indices is not None and len(boundary_indices) > 0:
            # Find edges on boundary (both nodes are boundary nodes)
            boundary_set = set(boundary_indices)
            for elem in elements:
                for i in range(3):
                    n1, n2 = elem[i], elem[(i+1) % 3]
                    if n1 in boundary_set and n2 in boundary_set:
                        edge = tuple(sorted([n1, n2]))
                        if edge not in [tuple(sorted(e)) for e in boundary_edges]:
                            boundary_edges.append([n1, n2])
        
        n_total_elements = n_elements + len(boundary_edges)
        
        f.write("$Elements\n")
        f.write(f"{n_total_elements}\n")
        
        elem_id = 1
        
        # Write boundary edges (type 1 = 2-node line)
        for edge in boundary_edges:
            # Format: id type num_tags physical_tag elementary_tag nodes...
            f.write(f"{elem_id} 1 2 1 1 {edge[0]+1} {edge[1]+1}\n")
            elem_id += 1
        
        # Write triangles (type 2 = 3-node triangle)
        for elem in elements:
            # Physical tag 2 = interior, elementary tag 2
            f.write(f"{elem_id} 2 2 2 2 {elem[0]+1} {elem[1]+1} {elem[2]+1}\n")
            elem_id += 1
        
        f.write("$EndElements\n")
        
        # Physical names (optional but helpful for gmsh GUI)
        f.write("$PhysicalNames\n")
        f.write("2\n")
        f.write(f"1 1 \"boundary\"\n")
        f.write(f"2 2 \"{domain_type}\"\n")
        f.write("$EndPhysicalNames\n")
    
    return filepath


def save_source_grid_npz(filepath: str, interior_points: np.ndarray,
                         metadata: dict = None):
    """
    Save source grid (inverse problem mesh) to numpy .npz format.
    
    Parameters
    ----------
    filepath : str
        Output path
    interior_points : np.ndarray
        Source grid points (n_points, 2)
    metadata : dict, optional
        Additional metadata (resolution, domain_type, etc.)
    """
    if not filepath.endswith('.npz'):
        filepath = filepath + '.npz'
    
    save_dict = {
        'interior_points': interior_points,
        'n_points': np.array([len(interior_points)]),
    }
    
    if metadata:
        import json
        save_dict['metadata'] = np.array([json.dumps(metadata)])
    
    np.savez(filepath, **save_dict)
    return filepath


def load_source_grid_npz(filepath: str) -> dict:
    """Load source grid from .npz format."""
    data = np.load(filepath, allow_pickle=True)
    result = {
        'interior_points': data['interior_points'],
        'n_points': int(data['n_points'][0]),
    }
    
    if 'metadata' in data:
        import json
        result['metadata'] = json.loads(str(data['metadata'][0]))
    else:
        result['metadata'] = {}
    
    return result


def save_source_grid_msh(filepath: str, interior_points: np.ndarray,
                         domain_type: str = 'source_grid'):
    """
    Save source grid as .msh file (points only, no elements).
    
    This creates a gmsh-compatible file with just nodes that can be 
    visualized in the gmsh GUI.
    
    Parameters
    ----------
    filepath : str
        Output path
    interior_points : np.ndarray
        Source grid points (n_points, 2)
    domain_type : str
        Name for physical group
    """
    if not filepath.endswith('.msh'):
        filepath = filepath + '.msh'
    
    n_points = len(interior_points)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes
        f.write("$Nodes\n")
        f.write(f"{n_points}\n")
        for i, pt in enumerate(interior_points):
            f.write(f"{i+1} {pt[0]:.15e} {pt[1]:.15e} 0.0\n")
        f.write("$EndNodes\n")
        
        # Elements - create point elements (type 15 = 1-node point)
        f.write("$Elements\n")
        f.write(f"{n_points}\n")
        for i in range(n_points):
            f.write(f"{i+1} 15 2 1 1 {i+1}\n")
        f.write("$EndElements\n")
        
        # Physical names
        f.write("$PhysicalNames\n")
        f.write("1\n")
        f.write(f"0 1 \"{domain_type}\"\n")
        f.write("$EndPhysicalNames\n")
    
    return filepath


def save_meshes(output_dir: str, 
                forward_mesh: dict = None,
                source_grid: np.ndarray = None,
                domain_type: str = 'generic',
                forward_resolution: float = None,
                source_resolution: float = None):
    """
    Save both forward mesh and source grid in both formats plus PNG visualization.
    
    Parameters
    ----------
    output_dir : str
        Directory to save meshes
    forward_mesh : dict
        Dict with keys: nodes, elements, boundary_indices, interior_indices
    source_grid : np.ndarray
        Source grid interior points (n_points, 2)
    domain_type : str
        Domain type name
    forward_resolution : float
        Forward mesh resolution (for metadata)
    source_resolution : float
        Source grid resolution (for metadata)
        
    Returns
    -------
    dict : Paths to saved files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save forward mesh
    if forward_mesh is not None:
        metadata = {
            'domain_type': domain_type,
            'resolution': forward_resolution,
            'n_nodes': len(forward_mesh['nodes']),
            'n_elements': len(forward_mesh['elements']),
            'n_boundary': len(forward_mesh['boundary_indices']),
            'n_interior': len(forward_mesh['interior_indices']),
        }
        
        # NPZ format
        npz_path = os.path.join(output_dir, f'forward_mesh_{domain_type}')
        save_mesh_npz(npz_path, 
                      forward_mesh['nodes'],
                      forward_mesh['elements'],
                      forward_mesh['boundary_indices'],
                      forward_mesh['interior_indices'],
                      metadata)
        saved_files['forward_mesh_npz'] = npz_path + '.npz'
        
        # MSH format
        msh_path = os.path.join(output_dir, f'forward_mesh_{domain_type}')
        save_mesh_msh(msh_path,
                      forward_mesh['nodes'],
                      forward_mesh['elements'],
                      forward_mesh['boundary_indices'],
                      domain_type)
        saved_files['forward_mesh_msh'] = msh_path + '.msh'
        
        # PNG visualization
        png_path = os.path.join(output_dir, f'forward_mesh_{domain_type}.png')
        plot_mesh(forward_mesh['nodes'], forward_mesh['elements'],
                  forward_mesh['boundary_indices'], forward_mesh['interior_indices'],
                  title=f"Forward Mesh: {domain_type}\n{metadata['n_nodes']} nodes, {metadata['n_elements']} elements, h={forward_resolution}",
                  save_path=png_path)
        saved_files['forward_mesh_png'] = png_path
    
    # Save source grid
    if source_grid is not None:
        metadata = {
            'domain_type': domain_type,
            'resolution': source_resolution,
            'n_points': len(source_grid),
        }
        
        # NPZ format
        npz_path = os.path.join(output_dir, f'source_grid_{domain_type}')
        save_source_grid_npz(npz_path, source_grid, metadata)
        saved_files['source_grid_npz'] = npz_path + '.npz'
        
        # MSH format
        msh_path = os.path.join(output_dir, f'source_grid_{domain_type}')
        save_source_grid_msh(msh_path, source_grid, f'{domain_type}_sources')
        saved_files['source_grid_msh'] = msh_path + '.msh'
        
        # PNG visualization
        png_path = os.path.join(output_dir, f'source_grid_{domain_type}.png')
        plot_source_grid(source_grid, 
                         boundary_points=forward_mesh['nodes'][forward_mesh['boundary_indices']] if forward_mesh else None,
                         title=f"Source Grid: {domain_type}\n{len(source_grid)} points, h={source_resolution}",
                         save_path=png_path)
        saved_files['source_grid_png'] = png_path
    
    # Combined visualization
    if forward_mesh is not None and source_grid is not None:
        png_path = os.path.join(output_dir, f'mesh_combined_{domain_type}.png')
        plot_mesh_and_sources(forward_mesh['nodes'], forward_mesh['elements'],
                              forward_mesh['boundary_indices'], source_grid,
                              title=f"Forward Mesh + Source Grid: {domain_type}",
                              save_path=png_path)
        saved_files['combined_png'] = png_path
    
    return saved_files


def plot_mesh(nodes: np.ndarray, elements: np.ndarray,
              boundary_indices: np.ndarray, interior_indices: np.ndarray,
              title: str = "Mesh", save_path: str = None, ax=None):
    """
    Plot triangular mesh with boundary and interior nodes highlighted.
    
    Parameters
    ----------
    nodes : np.ndarray
        Node coordinates (n_nodes, 2)
    elements : np.ndarray
        Triangle connectivity (n_elements, 3)
    boundary_indices : np.ndarray
        Indices of boundary nodes
    interior_indices : np.ndarray
        Indices of interior nodes
    title : str
        Plot title
    save_path : str
        If provided, save to this path
    ax : matplotlib.axes.Axes
        If provided, plot on this axes
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        own_fig = True
    else:
        own_fig = False
    
    # Plot triangles
    triangles = nodes[elements]
    pc = PolyCollection(triangles, facecolors='lightblue', 
                        edgecolors='gray', linewidths=0.3, alpha=0.7)
    ax.add_collection(pc)
    
    # Plot interior nodes
    if len(interior_indices) > 0:
        ax.plot(nodes[interior_indices, 0], nodes[interior_indices, 1], 
                'b.', markersize=2, label=f'Interior ({len(interior_indices)})')
    
    # Plot boundary nodes
    if len(boundary_indices) > 0:
        ax.plot(nodes[boundary_indices, 0], nodes[boundary_indices, 1], 
                'r.', markersize=4, label=f'Boundary ({len(boundary_indices)})')
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Auto-scale
    margin = 0.1 * max(nodes[:, 0].max() - nodes[:, 0].min(),
                       nodes[:, 1].max() - nodes[:, 1].min())
    ax.set_xlim(nodes[:, 0].min() - margin, nodes[:, 0].max() + margin)
    ax.set_ylim(nodes[:, 1].min() - margin, nodes[:, 1].max() + margin)
    
    if save_path and own_fig:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif own_fig:
        plt.close(fig)


def plot_source_grid(source_points: np.ndarray, boundary_points: np.ndarray = None,
                     title: str = "Source Grid", save_path: str = None, ax=None):
    """
    Plot source grid points with optional boundary.
    
    Parameters
    ----------
    source_points : np.ndarray
        Source grid points (n_points, 2)
    boundary_points : np.ndarray, optional
        Boundary points to show domain shape
    title : str
        Plot title
    save_path : str
        If provided, save to this path
    ax : matplotlib.axes.Axes
        If provided, plot on this axes
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        own_fig = True
    else:
        own_fig = False
    
    # Plot boundary if provided
    if boundary_points is not None:
        # Close the boundary loop
        boundary_closed = np.vstack([boundary_points, boundary_points[0:1]])
        ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 
                'k-', linewidth=1.5, label='Boundary')
        ax.fill(boundary_points[:, 0], boundary_points[:, 1], 
                color='lightyellow', alpha=0.3)
    
    # Plot source points
    ax.scatter(source_points[:, 0], source_points[:, 1], 
               c='green', s=20, alpha=0.7, label=f'Sources ({len(source_points)})')
    
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Auto-scale
    all_points = source_points if boundary_points is None else np.vstack([source_points, boundary_points])
    margin = 0.1 * max(all_points[:, 0].max() - all_points[:, 0].min(),
                       all_points[:, 1].max() - all_points[:, 1].min())
    ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    
    if save_path and own_fig:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    elif own_fig:
        plt.close(fig)


def plot_mesh_and_sources(nodes: np.ndarray, elements: np.ndarray,
                          boundary_indices: np.ndarray, source_points: np.ndarray,
                          title: str = "Mesh + Sources", save_path: str = None):
    """
    Plot forward mesh and source grid together for comparison.
    
    Parameters
    ----------
    nodes : np.ndarray
        Forward mesh nodes
    elements : np.ndarray
        Forward mesh triangles
    boundary_indices : np.ndarray
        Boundary node indices
    source_points : np.ndarray
        Source grid points
    title : str
        Plot title
    save_path : str
        If provided, save to this path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Forward mesh only
    ax = axes[0]
    triangles = nodes[elements]
    pc = PolyCollection(triangles, facecolors='lightblue', 
                        edgecolors='gray', linewidths=0.3, alpha=0.7)
    ax.add_collection(pc)
    ax.plot(nodes[boundary_indices, 0], nodes[boundary_indices, 1], 
            'r.', markersize=3)
    ax.set_aspect('equal')
    ax.set_title(f"Forward Mesh\n{len(nodes)} nodes, {len(elements)} triangles")
    ax.grid(True, alpha=0.3)
    
    # Middle: Source grid only
    ax = axes[1]
    boundary_points = nodes[boundary_indices]
    boundary_closed = np.vstack([boundary_points, boundary_points[0:1]])
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=1)
    ax.fill(boundary_points[:, 0], boundary_points[:, 1], color='lightyellow', alpha=0.3)
    ax.scatter(source_points[:, 0], source_points[:, 1], c='green', s=15, alpha=0.7)
    ax.set_aspect('equal')
    ax.set_title(f"Source Grid\n{len(source_points)} points")
    ax.grid(True, alpha=0.3)
    
    # Right: Combined overlay
    ax = axes[2]
    pc = PolyCollection(triangles, facecolors='lightblue', 
                        edgecolors='gray', linewidths=0.2, alpha=0.4)
    ax.add_collection(pc)
    ax.scatter(source_points[:, 0], source_points[:, 1], 
               c='green', s=25, alpha=0.8, zorder=5, label='Source grid')
    ax.plot(nodes[boundary_indices, 0], nodes[boundary_indices, 1], 
            'r.', markersize=3, label='Boundary')
    ax.set_aspect('equal')
    ax.set_title("Combined")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Set same limits for all
    all_points = np.vstack([nodes, source_points])
    margin = 0.1 * max(all_points[:, 0].max() - all_points[:, 0].min(),
                       all_points[:, 1].max() - all_points[:, 1].min())
    for ax in axes:
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.close(fig)
