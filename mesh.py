"""
Mesh Generation Module
======================

Provides uniform triangular meshes for:
1. FEM discretization (forward problem)
2. Source candidate locations (distributed inverse problem)

Both use gmsh for uniform triangular meshes.
"""

import numpy as np
from typing import Tuple, Optional


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
