"""
Mesh Utilities for Inverse Source Problem
==========================================
Handles mesh creation, loading, and manipulation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
    from mpi4py import MPI
    import dolfinx
    from dolfinx import mesh, io
    import gmsh
    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False
    print("Warning: FEniCSx not available. Some features will be limited.")


def create_disk_mesh(
    radius: float = 1.0,
    resolution: float = 0.05,
    center: Tuple[float, float] = (0.0, 0.0)
) -> "dolfinx.mesh.Mesh":
    """
    Create a circular disk mesh using Gmsh.
    
    Parameters
    ----------
    radius : float
        Radius of the disk
    resolution : float
        Target mesh element size
    center : tuple
        Center coordinates (x, y)
    
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
        The generated mesh
    cell_tags : MeshTags
        Cell markers
    facet_tags : MeshTags
        Facet markers (for boundary conditions)
    """
    if not HAS_FENICSX:
        raise ImportError("FEniCSx is required for mesh generation")
    
    gmsh.initialize()
    gmsh.model.add("disk")
    
    # Create disk
    disk = gmsh.model.occ.addDisk(center[0], center[1], 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Add physical groups for boundary and domain
    # Get boundary curves
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    
    gmsh.model.addPhysicalGroup(1, boundary_tags, 1)  # Boundary
    gmsh.model.setPhysicalName(1, 1, "Boundary")
    
    gmsh.model.addPhysicalGroup(2, [disk], 1)  # Domain
    gmsh.model.setPhysicalName(2, 1, "Domain")
    
    # Set mesh parameters
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.8)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution * 1.2)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    
    # Generate mesh
    gmsh.model.mesh.generate(2)
    
    # Convert to dolfinx
    msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        gmsh.model,
        MPI.COMM_WORLD,
        0,
        gdim=2
    )
    
    gmsh.finalize()
    
    return msh, cell_tags, facet_tags


def create_rectangle_mesh(
    width: float = 2.0,
    height: float = 2.0,
    resolution: float = 0.05,
    center: Tuple[float, float] = (0.0, 0.0)
) -> "dolfinx.mesh.Mesh":
    """
    Create a rectangular mesh using Gmsh.
    """
    if not HAS_FENICSX:
        raise ImportError("FEniCSx is required for mesh generation")
    
    gmsh.initialize()
    gmsh.model.add("rectangle")
    
    x0, y0 = center[0] - width/2, center[1] - height/2
    
    rect = gmsh.model.occ.addRectangle(x0, y0, 0, width, height)
    gmsh.model.occ.synchronize()
    
    # Physical groups
    boundary = gmsh.model.getBoundary([(2, rect)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    
    gmsh.model.addPhysicalGroup(1, boundary_tags, 1)
    gmsh.model.addPhysicalGroup(2, [rect], 1)
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.model.mesh.generate(2)
    
    msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
        gmsh.model,
        MPI.COMM_WORLD,
        0,
        gdim=2
    )
    
    gmsh.finalize()
    
    return msh, cell_tags, facet_tags


def load_mesh_from_file(filepath: str) -> "dolfinx.mesh.Mesh":
    """
    Load a mesh from a .msh file (Gmsh format).
    
    Parameters
    ----------
    filepath : str
        Path to the mesh file
    
    Returns
    -------
    mesh, cell_tags, facet_tags
    """
    if not HAS_FENICSX:
        raise ImportError("FEniCSx is required for mesh loading")
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Mesh file not found: {filepath}")
    
    if filepath.suffix == ".msh":
        gmsh.initialize()
        gmsh.open(str(filepath))
        
        msh, cell_tags, facet_tags = io.gmshio.model_to_mesh(
            gmsh.model,
            MPI.COMM_WORLD,
            0,
            gdim=2
        )
        
        gmsh.finalize()
        
    elif filepath.suffix == ".xdmf":
        with io.XDMFFile(MPI.COMM_WORLD, filepath, "r") as xdmf:
            msh = xdmf.read_mesh()
            cell_tags = None
            facet_tags = None
    else:
        raise ValueError(f"Unsupported mesh format: {filepath.suffix}")
    
    return msh, cell_tags, facet_tags


def save_mesh_to_file(
    msh: "dolfinx.mesh.Mesh",
    filepath: str,
    cell_tags=None,
    facet_tags=None
):
    """
    Save a mesh to file.
    
    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
        The mesh to save
    filepath : str
        Output path (.xdmf recommended)
    cell_tags, facet_tags : optional
        Mesh tags to save
    """
    filepath = Path(filepath)
    
    if filepath.suffix == ".xdmf":
        with io.XDMFFile(msh.comm, filepath, "w") as xdmf:
            xdmf.write_mesh(msh)
            if cell_tags is not None:
                xdmf.write_meshtags(cell_tags, msh.geometry)
    else:
        raise ValueError(f"Unsupported output format: {filepath.suffix}")


def get_mesh_info(msh: "dolfinx.mesh.Mesh") -> dict:
    """
    Get information about a mesh.
    
    Returns
    -------
    dict with keys: num_vertices, num_cells, bounds, etc.
    """
    coords = msh.geometry.x
    
    info = {
        "num_vertices": msh.topology.index_map(0).size_local,
        "num_cells": msh.topology.index_map(msh.topology.dim).size_local,
        "gdim": msh.geometry.dim,
        "tdim": msh.topology.dim,
        "bounds": {
            "x_min": coords[:, 0].min(),
            "x_max": coords[:, 0].max(),
            "y_min": coords[:, 1].min(),
            "y_max": coords[:, 1].max(),
        },
        "cell_type": str(msh.topology.cell_type),
    }
    
    return info


def plot_mesh(
    msh: "dolfinx.mesh.Mesh",
    sources: Optional[list] = None,
    save_path: Optional[str] = None,
    show_edges: bool = True
):
    """
    Plot the mesh with optional source locations.
    """
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    
    coords = msh.geometry.x
    
    # Get cell connectivity
    cells = msh.topology.connectivity(msh.topology.dim, 0)
    triangles = []
    for i in range(cells.num_nodes):
        triangles.append(cells.links(i))
    triangles = np.array(triangles)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if show_edges:
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.5)
    
    # Plot boundary
    ax.plot(coords[:, 0], coords[:, 1], '.', markersize=1, alpha=0.3)
    
    # Plot sources if provided
    if sources is not None:
        for (x, y), intensity in sources:
            color = 'red' if intensity > 0 else 'blue'
            marker = '+' if intensity > 0 else '*'
            label = 'Positive Source' if intensity > 0 else 'Negative Source'
            ax.plot(x, y, marker, color=color, markersize=12, 
                   markeredgewidth=2, label=label)
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mesh of the Unit Disk with Source Locations')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing mesh utilities...")
    
    # Create disk mesh
    msh, cell_tags, facet_tags = create_disk_mesh(radius=1.0, resolution=0.05)
    
    # Get info
    info = get_mesh_info(msh)
    print("\nMesh info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test plot
    sources = [
        ((-0.3, 0.4), 1.0),
        ((0.5, -0.3), -1.0),
    ]
    plot_mesh(msh, sources, save_path="test_mesh.png")
    
    print("\nDone!")
