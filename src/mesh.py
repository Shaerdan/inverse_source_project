"""
Mesh Generation Module
======================

Provides uniform triangular meshes for:
1. FEM discretization (forward problem)
2. Source candidate locations (distributed inverse problem)

Both use gmsh for uniform triangular meshes.
"""

import numpy as np
from typing import Tuple, Optional, List


def create_disk_mesh(resolution: float = 0.1, radius: float = 1.0, 
                     interior_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform triangular mesh of the disk using gmsh.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size (smaller = finer mesh)
    radius : float
        Disk radius
    interior_only : bool
        If True, only return interior points (for source grid)
        
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
    """
    try:
        return _create_mesh_gmsh(resolution, radius)
    except Exception as e:
        print(f"Note: gmsh unavailable ({e}), using fallback mesh")
        return _create_mesh_fallback(resolution, radius)


def _create_mesh_gmsh(resolution: float, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create mesh using gmsh."""
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
    gmsh.model.add("disk")
    
    # Create disk geometry
    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Set mesh size for uniform elements
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.8)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    n_nodes = len(node_tags)
    
    # Create mapping from gmsh tags to sequential indices
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    
    # Reshape coordinates
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
                n1 = tag_to_idx[int(node_tags_flat[3*j])]
                n2 = tag_to_idx[int(node_tags_flat[3*j + 1])]
                n3 = tag_to_idx[int(node_tags_flat[3*j + 2])]
                elements.append([n1, n2, n3])
    
    elements = np.array(elements)
    
    gmsh.finalize()
    
    # Classify nodes
    radii = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    boundary_indices = np.where(radii > 0.99 * radius)[0]
    interior_indices = np.where(radii <= 0.99 * radius)[0]
    
    return nodes, elements, boundary_indices, interior_indices


def _create_mesh_fallback(resolution: float, radius: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback mesh using sunflower pattern + Delaunay.
    Produces quasi-uniform distribution without gmsh.
    """
    from scipy.spatial import Delaunay
    
    # Estimate number of points
    area = np.pi * radius**2
    n_total = int(area / (resolution**2 * 0.5))  # rough estimate
    n_boundary = int(2 * np.pi * radius / resolution)
    n_interior = n_total - n_boundary
    
    # Boundary points (uniform on circle)
    theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    boundary_pts = np.column_stack([
        radius * np.cos(theta_boundary),
        radius * np.sin(theta_boundary)
    ])
    
    # Interior points using sunflower/Vogel pattern (uniform area distribution)
    indices = np.arange(1, n_interior + 1)
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees
    
    r_int = radius * 0.98 * np.sqrt(indices / n_interior)  # sqrt for uniform area
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
    
    return nodes, elements, boundary_indices, interior_indices


def get_source_grid(resolution: float = 0.15, radius: float = 0.9) -> np.ndarray:
    """
    Get interior points for source candidate locations.
    
    Convenience function that returns just the interior node positions.
    
    Parameters
    ----------
    resolution : float
        Grid resolution (larger = coarser = fewer candidates)
    radius : float
        Maximum radius for source locations (< 1.0 to stay inside domain)
        
    Returns
    -------
    points : array, shape (M, 2)
        Interior point coordinates for source candidates
    """
    nodes, elements, boundary_idx, interior_idx = create_disk_mesh(resolution, radius=1.0)
    
    # Filter to requested radius
    interior_points = nodes[interior_idx]
    radii = np.sqrt(interior_points[:, 0]**2 + interior_points[:, 1]**2)
    mask = radii < radius
    
    return interior_points[mask]


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

def create_ellipse_mesh(a: float = 2.0, b: float = 1.0, resolution: float = 0.1
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        
    Returns
    -------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    boundary_indices : array
    interior_indices : array
    """
    try:
        return _create_ellipse_mesh_gmsh(a, b, resolution)
    except Exception as e:
        print(f"Note: gmsh unavailable ({e}), using fallback ellipse mesh")
        return _create_ellipse_mesh_fallback(a, b, resolution)


def _create_ellipse_mesh_gmsh(a: float, b: float, resolution: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create ellipse mesh using gmsh."""
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ellipse")
    
    # Create ellipse (disk with different radii)
    ellipse = gmsh.model.occ.addDisk(0, 0, 0, a, b)
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.8)
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
    
    # Classify nodes (on ellipse boundary if (x/a)² + (y/b)² ≈ 1)
    ellipse_param = (nodes[:, 0]/a)**2 + (nodes[:, 1]/b)**2
    boundary_indices = np.where(ellipse_param > 0.98)[0]
    interior_indices = np.where(ellipse_param <= 0.98)[0]
    
    return nodes, elements, boundary_indices, interior_indices


def _create_ellipse_mesh_fallback(a: float, b: float, resolution: float
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback ellipse mesh using sunflower pattern + Delaunay."""
    from scipy.spatial import Delaunay
    
    # Estimate number of points
    area = np.pi * a * b
    n_total = int(area / (resolution**2 * 0.5))
    perimeter = np.pi * (3*(a+b) - np.sqrt((3*a+b)*(a+3*b)))  # Ramanujan approx
    n_boundary = max(int(perimeter / resolution), 20)
    n_interior = n_total - n_boundary
    
    # Boundary points
    theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    boundary_pts = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    
    # Interior points using transformed sunflower pattern
    indices = np.arange(1, n_interior + 1)
    golden_angle = np.pi * (3 - np.sqrt(5))
    
    # Generate in unit disk, then scale to ellipse
    r = 0.95 * np.sqrt(indices / n_interior)
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
    
    return nodes, elements, boundary_indices, interior_indices


def get_ellipse_source_grid(a: float = 2.0, b: float = 1.0, 
                            resolution: float = 0.15, margin: float = 0.1
                            ) -> np.ndarray:
    """
    Get interior points for source candidates in an ellipse.
    
    Parameters
    ----------
    a, b : float
        Semi-axes of ellipse
    resolution : float
        Grid resolution
    margin : float
        Keep sources this fraction away from boundary
        
    Returns
    -------
    points : array, shape (M, 2)
    """
    nodes, _, _, interior_idx = create_ellipse_mesh(a, b, resolution)
    interior_points = nodes[interior_idx]
    
    # Filter to margin
    ellipse_param = (interior_points[:, 0]/a)**2 + (interior_points[:, 1]/b)**2
    mask = ellipse_param < (1 - margin)**2
    
    return interior_points[mask]


# =============================================================================
# POLYGON MESH GENERATION
# =============================================================================

def create_polygon_mesh(vertices: List[Tuple[float, float]], resolution: float = 0.1
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform triangular mesh of a polygon.
    
    Parameters
    ----------
    vertices : list of (x, y) tuples
        Polygon vertices in order (CCW for standard orientation)
    resolution : float
        Target mesh element size
        
    Returns
    -------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    boundary_indices : array
    interior_indices : array
    
    Examples
    --------
    >>> # Unit square
    >>> verts = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    >>> nodes, elems, b_idx, i_idx = create_polygon_mesh(verts, resolution=0.2)
    
    >>> # L-shaped domain
    >>> verts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    >>> nodes, elems, b_idx, i_idx = create_polygon_mesh(verts, resolution=0.15)
    """
    try:
        return _create_polygon_mesh_gmsh(vertices, resolution)
    except Exception as e:
        print(f"Note: gmsh unavailable ({e}), using fallback polygon mesh")
        return _create_polygon_mesh_fallback(vertices, resolution)


def _create_polygon_mesh_gmsh(vertices: List[Tuple[float, float]], resolution: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create polygon mesh using gmsh."""
    import gmsh
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("polygon")
    
    # Create polygon
    n_verts = len(vertices)
    points = []
    for x, y in vertices:
        points.append(gmsh.model.occ.addPoint(x, y, 0, resolution))
    
    lines = []
    for i in range(n_verts):
        lines.append(gmsh.model.occ.addLine(points[i], points[(i+1) % n_verts]))
    
    curve_loop = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([curve_loop])
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.8)
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
    vertices_arr = np.array(vertices)
    boundary_indices = _find_polygon_boundary_nodes(nodes, vertices_arr, tol=resolution*0.5)
    interior_indices = np.setdiff1d(np.arange(len(nodes)), boundary_indices)
    
    return nodes, elements, boundary_indices, interior_indices


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


def _create_polygon_mesh_fallback(vertices: List[Tuple[float, float]], resolution: float
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback polygon mesh using boundary sampling + Delaunay."""
    from scipy.spatial import Delaunay
    
    vertices_arr = np.array(vertices)
    n_verts = len(vertices)
    
    # Sample boundary edges
    boundary_pts = []
    for i in range(n_verts):
        p1 = vertices_arr[i]
        p2 = vertices_arr[(i + 1) % n_verts]
        edge_len = np.linalg.norm(p2 - p1)
        n_edge_pts = max(int(edge_len / resolution), 2)
        
        for j in range(n_edge_pts):
            t = j / n_edge_pts
            boundary_pts.append(p1 + t * (p2 - p1))
    
    boundary_pts = np.array(boundary_pts)
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
    
    # Remove interior points too close to boundary
    if len(interior_pts) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(boundary_pts)
        dists, _ = tree.query(interior_pts)
        interior_pts = interior_pts[dists > resolution * 0.3]
    
    nodes = np.vstack([boundary_pts, interior_pts])
    
    # Triangulate
    if len(nodes) < 3:
        return nodes, np.array([]), np.arange(n_boundary), np.array([])
    
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # Remove triangles outside polygon
    centroids = nodes[elements].mean(axis=1)
    inside = _points_in_polygon(centroids, vertices_arr)
    elements = elements[inside]
    
    boundary_indices = np.arange(n_boundary)
    interior_indices = np.arange(n_boundary, len(nodes))
    
    return nodes, elements, boundary_indices, interior_indices


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
                            resolution: float = 0.15, margin: float = 0.1
                            ) -> np.ndarray:
    """
    Get interior points for source candidates in a polygon.
    
    Parameters
    ----------
    vertices : list of (x, y)
        Polygon vertices
    resolution : float
        Grid resolution
    margin : float
        Keep sources this distance away from boundary
        
    Returns
    -------
    points : array, shape (M, 2)
    """
    nodes, _, boundary_idx, interior_idx = create_polygon_mesh(vertices, resolution)
    interior_points = nodes[interior_idx]
    
    if len(interior_points) == 0:
        return interior_points
    
    # Filter points too close to boundary
    boundary_pts = nodes[boundary_idx]
    if len(boundary_pts) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(boundary_pts)
        dists, _ = tree.query(interior_points)
        mask = dists > margin
        return interior_points[mask]
    
    return interior_points


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
