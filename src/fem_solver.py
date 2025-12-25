"""
Finite Element Method Solver for Poisson Equation with Point Sources
=====================================================================

This module implements FEM-based forward solver using DOLFINx.
Requires DOLFINx 0.8+ (uses updated API).

Two source handling methods:
    - 'snap': Assign source to nearest mesh node (approximate, fast)
    - 'interpolate': Use barycentric interpolation (exact, slightly slower)
"""

import numpy as np
from typing import List, Tuple, Optional

# Check for DOLFINx availability
try:
    import dolfinx
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    import ufl
    from mpi4py import MPI
    import gmsh
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False
    print("Warning: DOLFINx not available. FEM solver disabled.")


def create_disk_mesh(resolution: float = 0.05, radius: float = 1.0):
    """
    Create a triangular mesh of the unit disk using Gmsh.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size
    radius : float
        Disk radius
        
    Returns
    -------
    mesh : dolfinx.mesh.Mesh
    cell_tags : MeshTags
    facet_tags : MeshTags
    """
    if not HAS_DOLFINX:
        raise RuntimeError("DOLFINx not available")
    
    gmsh.initialize()
    gmsh.model.add("disk")
    
    # Create disk geometry
    disk = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
    gmsh.model.occ.synchronize()
    
    # Set mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.5)
    
    # Mark boundary
    gmsh.model.addPhysicalGroup(2, [disk], 1)
    gmsh.model.setPhysicalName(2, 1, "Domain")
    
    boundary = gmsh.model.getBoundary([(2, disk)], oriented=False)
    boundary_tags = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(1, boundary_tags, 2)
    gmsh.model.setPhysicalName(1, 2, "Boundary")
    
    gmsh.model.mesh.generate(2)
    
    # Convert to DOLFINx mesh
    msh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2
    )
    
    gmsh.finalize()
    return msh, cell_tags, facet_tags


def find_containing_cell(msh, point: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the mesh cell containing a point and compute barycentric coordinates.
    
    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    point : array, shape (2,)
    
    Returns
    -------
    cell_idx : int
        Index of containing cell (-1 if not found)
    bary_coords : array, shape (3,)
        Barycentric coordinates in the cell
    """
    geom = msh.geometry.x
    cells = msh.topology.connectivity(2, 0)
    
    x, y = point[0], point[1]
    
    for cell_idx in range(cells.num_nodes):
        cell_nodes = cells.links(cell_idx)
        x1, y1 = geom[cell_nodes[0], 0], geom[cell_nodes[0], 1]
        x2, y2 = geom[cell_nodes[1], 0], geom[cell_nodes[1], 1]
        x3, y3 = geom[cell_nodes[2], 0], geom[cell_nodes[2], 1]
        
        # Compute barycentric coordinates
        det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if abs(det) < 1e-14:
            continue
        
        lam1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
        lam2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
        lam3 = 1 - lam1 - lam2
        
        # Check if point is inside triangle
        if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
            return cell_idx, np.array([lam1, lam2, lam3])
    
    return -1, np.array([0, 0, 0])


def solve_poisson_zero_neumann(msh, sources: List[Tuple[Tuple[float, float], float]],
                                method: str = 'interpolate') -> fem.Function:
    """
    Solve Poisson equation with point sources and zero Neumann BC.
    
    -Δu = Σ qₖ δ(x - ξₖ)  in Ω
    ∂u/∂n = 0             on ∂Ω
    
    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    sources : list of ((x, y), q) tuples
    method : str, 'snap' or 'interpolate'
    
    Returns
    -------
    u : dolfinx.fem.Function
        FEM solution
    """
    if not HAS_DOLFINX:
        raise RuntimeError("DOLFINx not available")
    
    # Create function space (P1 elements)
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form (stiffness matrix)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Build RHS vector
    n_dofs = V.dofmap.index_map.size_local
    b_array = np.zeros(n_dofs)
    
    geom = msh.geometry.x
    
    if method == 'snap':
        # Simple: assign source to nearest node
        for (xi_x, xi_y), q in sources:
            xi = np.array([xi_x, xi_y])
            distances = np.sqrt((geom[:, 0] - xi_x)**2 + (geom[:, 1] - xi_y)**2)
            nearest = np.argmin(distances)
            b_array[nearest] += q
    
    elif method == 'interpolate':
        # Correct: use barycentric interpolation
        cells = msh.topology.connectivity(2, 0)
        
        for (xi_x, xi_y), q in sources:
            cell_idx, bary = find_containing_cell(msh, np.array([xi_x, xi_y]))
            
            if cell_idx < 0:
                # Fall back to snap if outside mesh
                distances = np.sqrt((geom[:, 0] - xi_x)**2 + (geom[:, 1] - xi_y)**2)
                nearest = np.argmin(distances)
                b_array[nearest] += q
            else:
                cell_nodes = cells.links(cell_idx)
                for i, node in enumerate(cell_nodes):
                    b_array[node] += q * bary[i]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create RHS as a Function
    f = fem.Function(V)
    f.x.array[:] = 0
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Assemble system
    A = fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()
    
    b = fem.petsc.create_vector(fem.form(L))
    b.array[:] = b_array
    
    # Solve (using PETSc)
    from petsc4py import PETSc
    
    u_h = fem.Function(V)
    
    solver = PETSc.KSP().create(msh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10)
    solver.solve(b, u_h.vector)
    
    # Enforce zero mean
    u_h.x.array[:] -= np.mean(u_h.x.array)
    
    return u_h


def get_boundary_values(u_h: fem.Function, radius_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract solution values at boundary nodes.
    
    Parameters
    ----------
    u_h : dolfinx.fem.Function
        FEM solution
    radius_threshold : float
        Nodes with |x| > threshold are considered boundary
        
    Returns
    -------
    theta : array
        Angular positions of boundary nodes
    u_boundary : array
        Solution values at boundary nodes
    """
    msh = u_h.function_space.mesh
    geom = msh.geometry.x
    
    # Find boundary nodes
    radii = np.sqrt(geom[:, 0]**2 + geom[:, 1]**2)
    boundary_mask = radii > radius_threshold
    boundary_indices = np.where(boundary_mask)[0]
    
    # Get angles and values
    theta = np.arctan2(geom[boundary_indices, 1], geom[boundary_indices, 0])
    u_values = u_h.x.array[boundary_indices]
    
    # Sort by angle
    sort_idx = np.argsort(theta)
    
    return theta[sort_idx], u_values[sort_idx]


# =============================================================================
# FEM-BASED LINEAR INVERSE SOLVER
# =============================================================================

class FEMLinearInverseSolver:
    """
    Linear inverse solver using FEM Green's matrix.
    
    Computes G[i,j] = u_boundary[i] when unit source at interior node j.
    """
    
    def __init__(self, msh, boundary_threshold: float = 0.95, interior_threshold: float = 0.9):
        if not HAS_DOLFINX:
            raise RuntimeError("DOLFINx not available")
        
        self.msh = msh
        self.geom = msh.geometry.x
        
        # Identify boundary and interior nodes
        radii = np.sqrt(self.geom[:, 0]**2 + self.geom[:, 1]**2)
        self.boundary_indices = np.where(radii > boundary_threshold)[0]
        self.interior_indices = np.where(radii < interior_threshold)[0]
        
        self.n_boundary = len(self.boundary_indices)
        self.n_interior = len(self.interior_indices)
        
        self.G = None
    
    def build_greens_matrix(self, method: str = 'interpolate'):
        """Build the Green's matrix by solving N forward problems."""
        print(f"Building FEM Green's matrix ({self.n_boundary} x {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j, node_idx in enumerate(self.interior_indices):
            if j % 50 == 0:
                print(f"  Column {j}/{self.n_interior}")
            
            x, y = self.geom[node_idx, 0], self.geom[node_idx, 1]
            sources = [((x, y), 1.0), ((0, 0), -1.0)]  # Add sink at origin
            
            u_h = solve_poisson_zero_neumann(self.msh, sources, method=method)
            
            # Extract boundary values
            self.G[:, j] = u_h.x.array[self.boundary_indices]
        
        # Zero mean columns
        self.G = self.G - np.mean(self.G, axis=0, keepdims=True)
        print("Done.")
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        """Solve with L1 regularization."""
        if self.G is None:
            self.build_greens_matrix()
        
        u = u_measured - np.mean(u_measured)
        q = np.zeros(self.n_interior)
        eps = 1e-4
        
        GtG = self.G.T @ self.G
        Gtu = self.G.T @ u
        
        for _ in range(max_iter):
            W = np.diag(1.0 / (np.abs(q) + eps))
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            if np.linalg.norm(q_new - q) < 1e-6:
                break
            q = q_new
        
        return q - np.mean(q)
    
    def get_interior_positions(self) -> np.ndarray:
        """Return positions of interior nodes."""
        return self.geom[self.interior_indices, :2]
