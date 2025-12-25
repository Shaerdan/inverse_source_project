"""
Finite Element Method Solver for Poisson Equation with Point Sources
=====================================================================

This module implements FEM-based forward and inverse solvers.

Formulations:
    - FEMForwardSolver: Forward problem with continuous source positions
    - FEMLinearInverseSolver: Linear inverse (sources on grid)
    - FEMNonlinearInverseSolver: Nonlinear inverse (continuous source positions)

Source Handling Methods:
    - 'snap': Assign source to nearest mesh node (approximate, fast)
    - 'interpolate': Use barycentric interpolation (exact location)

Requires DOLFINx 0.8+ for full functionality.
Falls back to scipy-based solver if DOLFINx unavailable.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Optional
from dataclasses import dataclass


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
    print("Note: DOLFINx not available. Using scipy-based FEM solver.")


# =============================================================================
# DATA CLASSES (shared with BEM for consistency)
# =============================================================================

@dataclass
class Source:
    """Represents a point source."""
    x: float
    y: float
    intensity: float
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_tuple(self) -> Tuple[Tuple[float, float], float]:
        return ((self.x, self.y), self.intensity)


@dataclass 
class InverseResult:
    """Results from inverse solver."""
    sources: List[Source]
    residual: float
    success: bool
    message: str
    iterations: int
    history: List[float]


# =============================================================================
# MESH UTILITIES
# =============================================================================

def create_disk_mesh_scipy(n_radial: int = 10, n_angular: int = 20, radius: float = 1.0):
    """
    Create a simple triangular mesh of the unit disk using scipy.
    
    Returns nodes, elements, boundary_indices for scipy-based FEM.
    """
    from scipy.spatial import Delaunay
    
    # Create nodes
    nodes = [(0.0, 0.0)]  # Center
    
    for i in range(1, n_radial + 1):
        r = radius * i / n_radial
        n_theta = max(6, int(n_angular * i / n_radial))
        for j in range(n_theta):
            theta = 2 * np.pi * j / n_theta
            nodes.append((r * np.cos(theta), r * np.sin(theta)))
    
    nodes = np.array(nodes)
    
    # Triangulate
    tri = Delaunay(nodes)
    elements = tri.simplices
    
    # Find boundary nodes
    radii = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    boundary_indices = np.where(radii > 0.95 * radius)[0]
    
    return nodes, elements, boundary_indices


def create_disk_mesh_dolfinx(resolution: float = 0.05, radius: float = 1.0):
    """
    Create a triangular mesh of the unit disk using Gmsh + DOLFINx.
    
    Parameters
    ----------
    resolution : float
        Target mesh element size
    radius : float
        Disk radius
        
    Returns
    -------
    msh : dolfinx.mesh.Mesh
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


def find_containing_cell(nodes: np.ndarray, elements: np.ndarray, 
                         point: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the mesh cell containing a point and compute barycentric coordinates.
    
    Parameters
    ----------
    nodes : array, shape (N, 2)
        Mesh node coordinates
    elements : array, shape (M, 3)
        Triangle connectivity
    point : array, shape (2,)
        Point to locate
    
    Returns
    -------
    cell_idx : int
        Index of containing cell (-1 if not found)
    bary_coords : array, shape (3,)
        Barycentric coordinates in the cell
    """
    x, y = point[0], point[1]
    
    for cell_idx, cell in enumerate(elements):
        x1, y1 = nodes[cell[0]]
        x2, y2 = nodes[cell[1]]
        x3, y3 = nodes[cell[2]]
        
        # Compute barycentric coordinates
        det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if abs(det) < 1e-14:
            continue
        
        lam1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
        lam2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
        lam3 = 1 - lam1 - lam2
        
        # Check if point is inside triangle (with tolerance)
        if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
            return cell_idx, np.array([lam1, lam2, lam3])
    
    return -1, np.array([0, 0, 0])


# =============================================================================
# SCIPY-BASED FEM SOLVER (no DOLFINx required)
# =============================================================================

def assemble_stiffness_matrix(nodes: np.ndarray, elements: np.ndarray) -> csr_matrix:
    """
    Assemble FEM stiffness matrix for P1 elements.
    
    Uses the formula for linear triangular elements:
    K_local = (1/4A) * [b_i*b_j + c_i*c_j] for i,j in element
    where b_i = y_j - y_k, c_i = x_k - x_j (cyclic)
    """
    n_nodes = len(nodes)
    row, col, data = [], [], []
    
    for element in elements:
        # Get element node coordinates
        coords = nodes[element]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        # Element area
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        if area < 1e-14:
            continue
        
        # Shape function gradients (constant for P1)
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        # Local stiffness matrix
        K_local = np.outer(b, b) + np.outer(c, c)
        K_local /= (4 * area)
        
        # Assemble into global
        for i in range(3):
            for j in range(3):
                row.append(element[i])
                col.append(element[j])
                data.append(K_local[i, j])
    
    K = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    return K


def solve_poisson_scipy(nodes: np.ndarray, elements: np.ndarray,
                        sources: List[Tuple[Tuple[float, float], float]],
                        method: str = 'interpolate') -> np.ndarray:
    """
    Solve Poisson equation with point sources using scipy (no DOLFINx).
    
    -Δu = Σ qₖ δ(x - ξₖ)  in Ω
    ∂u/∂n = 0             on ∂Ω
    
    Parameters
    ----------
    nodes : array, shape (N, 2)
    elements : array, shape (M, 3)
    sources : list of ((x, y), q) tuples
    method : str, 'snap' or 'interpolate'
    
    Returns
    -------
    u : array, shape (N,)
        Solution at mesh nodes
    """
    n_nodes = len(nodes)
    
    # Assemble stiffness matrix
    K = assemble_stiffness_matrix(nodes, elements)
    
    # Build RHS
    f = np.zeros(n_nodes)
    
    if method == 'snap':
        for (xi_x, xi_y), q in sources:
            distances = np.sqrt((nodes[:, 0] - xi_x)**2 + (nodes[:, 1] - xi_y)**2)
            nearest = np.argmin(distances)
            f[nearest] += q
    
    elif method == 'interpolate':
        for (xi_x, xi_y), q in sources:
            cell_idx, bary = find_containing_cell(nodes, elements, np.array([xi_x, xi_y]))
            
            if cell_idx < 0:
                # Fall back to snap
                distances = np.sqrt((nodes[:, 0] - xi_x)**2 + (nodes[:, 1] - xi_y)**2)
                nearest = np.argmin(distances)
                f[nearest] += q
            else:
                cell_nodes = elements[cell_idx]
                for i, node in enumerate(cell_nodes):
                    f[node] += q * bary[i]
    
    # Regularize for Neumann problem (fix mean = 0)
    # Add regularization: K + εI
    eps = 1e-10
    K_reg = K + eps * diags([1.0] * n_nodes)
    
    # Solve
    u = spsolve(K_reg, f)
    
    # Enforce zero mean
    u = u - np.mean(u)
    
    return u


# =============================================================================
# DOLFINx-BASED FEM SOLVER
# =============================================================================

def solve_poisson_dolfinx(msh, sources: List[Tuple[Tuple[float, float], float]],
                          method: str = 'interpolate'):
    """
    Solve Poisson equation with DOLFINx.
    
    Returns DOLFINx Function object.
    """
    if not HAS_DOLFINX:
        raise RuntimeError("DOLFINx not available")
    
    from petsc4py import PETSc
    
    # Create function space
    V = fem.functionspace(msh, ("Lagrange", 1))
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Bilinear form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    
    # Build RHS
    n_dofs = V.dofmap.index_map.size_local
    b_array = np.zeros(n_dofs)
    
    geom = msh.geometry.x
    cells = msh.topology.connectivity(2, 0)
    
    if method == 'snap':
        for (xi_x, xi_y), q in sources:
            distances = np.sqrt((geom[:, 0] - xi_x)**2 + (geom[:, 1] - xi_y)**2)
            nearest = np.argmin(distances)
            b_array[nearest] += q
    
    elif method == 'interpolate':
        for (xi_x, xi_y), q in sources:
            # Find containing cell
            cell_idx = -1
            bary = None
            
            for c in range(cells.num_nodes):
                cell_nodes = cells.links(c)
                x1, y1 = geom[cell_nodes[0], 0], geom[cell_nodes[0], 1]
                x2, y2 = geom[cell_nodes[1], 0], geom[cell_nodes[1], 1]
                x3, y3 = geom[cell_nodes[2], 0], geom[cell_nodes[2], 1]
                
                det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
                if abs(det) < 1e-14:
                    continue
                
                lam1 = ((y2 - y3)*(xi_x - x3) + (x3 - x2)*(xi_y - y3)) / det
                lam2 = ((y3 - y1)*(xi_x - x3) + (x1 - x3)*(xi_y - y3)) / det
                lam3 = 1 - lam1 - lam2
                
                if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
                    cell_idx = c
                    bary = np.array([lam1, lam2, lam3])
                    break
            
            if cell_idx < 0:
                distances = np.sqrt((geom[:, 0] - xi_x)**2 + (geom[:, 1] - xi_y)**2)
                nearest = np.argmin(distances)
                b_array[nearest] += q
            else:
                cell_nodes = cells.links(cell_idx)
                for i, node in enumerate(cell_nodes):
                    b_array[node] += q * bary[i]
    
    # Create RHS Function
    f = fem.Function(V)
    f.x.array[:] = 0
    L = ufl.inner(f, v) * ufl.dx
    
    # Assemble
    A = fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()
    
    b = fem.petsc.create_vector(fem.form(L))
    b.array[:] = b_array
    
    # Solve
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


# =============================================================================
# FEM FORWARD SOLVER CLASS
# =============================================================================

class FEMForwardSolver:
    """
    FEM Forward solver for Poisson equation with point sources.
    
    Handles truly continuous source positions via barycentric interpolation.
    Works with or without DOLFINx.
    
    Parameters
    ----------
    n_radial : int
        Number of radial mesh divisions (scipy mesh)
    n_angular : int
        Number of angular mesh divisions (scipy mesh)
    resolution : float
        Mesh resolution for DOLFINx mesh
    use_dolfinx : bool
        Use DOLFINx if available
    """
    
    def __init__(self, n_radial: int = 15, n_angular: int = 30, 
                 resolution: float = 0.05, use_dolfinx: bool = False):
        
        self.use_dolfinx = use_dolfinx and HAS_DOLFINX
        
        if self.use_dolfinx:
            self.msh, _, _ = create_disk_mesh_dolfinx(resolution)
            self.nodes = self.msh.geometry.x[:, :2]
            radii = np.sqrt(self.nodes[:, 0]**2 + self.nodes[:, 1]**2)
            self.boundary_indices = np.where(radii > 0.95)[0]
        else:
            self.nodes, self.elements, self.boundary_indices = create_disk_mesh_scipy(n_radial, n_angular)
        
        # Boundary info
        boundary_points = self.nodes[self.boundary_indices]
        self.theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.boundary_indices = self.boundary_indices[sort_idx]
        self.n_boundary = len(self.boundary_indices)
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]], 
              method: str = 'interpolate') -> np.ndarray:
        """
        Compute boundary values for given sources.
        
        Parameters
        ----------
        sources : list of ((x, y), q) tuples
            Point sources (positions can be continuous!)
        method : str
            'snap' or 'interpolate'
            
        Returns
        -------
        u_boundary : array
            Solution values at boundary points
        """
        # Check compatibility condition
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6e} ≠ 0")
        
        if self.use_dolfinx:
            u_h = solve_poisson_dolfinx(self.msh, sources, method)
            u = u_h.x.array[self.boundary_indices]
        else:
            u_full = solve_poisson_scipy(self.nodes, self.elements, sources, method)
            u = u_full[self.boundary_indices]
        
        return u - np.mean(u)
    
    def solve_full(self, sources: List[Tuple[Tuple[float, float], float]],
                   method: str = 'interpolate') -> np.ndarray:
        """Return solution at all mesh nodes."""
        if self.use_dolfinx:
            u_h = solve_poisson_dolfinx(self.msh, sources, method)
            return u_h.x.array
        else:
            return solve_poisson_scipy(self.nodes, self.elements, sources, method)


# =============================================================================
# FEM LINEAR INVERSE SOLVER (Grid-Based)
# =============================================================================

class FEMLinearInverseSolver:
    """
    Linear inverse solver using FEM Green's matrix.
    
    Sources are constrained to interior mesh nodes.
    Supports L1, L2, and TV regularization.
    
    Parameters
    ----------
    n_radial : int
        Mesh radial divisions
    n_angular : int
        Mesh angular divisions
    interior_threshold : float
        Nodes with |x| < threshold are candidate source locations
    """
    
    def __init__(self, n_radial: int = 15, n_angular: int = 30,
                 interior_threshold: float = 0.85):
        
        self.nodes, self.elements, self.boundary_indices = create_disk_mesh_scipy(n_radial, n_angular)
        
        # Identify interior nodes
        radii = np.sqrt(self.nodes[:, 0]**2 + self.nodes[:, 1]**2)
        self.interior_indices = np.where(radii < interior_threshold)[0]
        self.n_interior = len(self.interior_indices)
        
        # Sort boundary by angle
        boundary_points = self.nodes[self.boundary_indices]
        theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        sort_idx = np.argsort(theta)
        self.boundary_indices = self.boundary_indices[sort_idx]
        self.theta = theta[sort_idx]
        self.n_boundary = len(self.boundary_indices)
        
        self.interior_points = self.nodes[self.interior_indices]
        self.G = None
    
    def build_greens_matrix(self, method: str = 'interpolate'):
        """
        Build the Green's matrix by solving N forward problems.
        
        G[i, j] = u(boundary_point_i) when unit source at interior_point_j
        """
        print(f"Building FEM Green's matrix ({self.n_boundary} x {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j in range(self.n_interior):
            if j % 50 == 0:
                print(f"  Column {j}/{self.n_interior}")
            
            # Unit source at interior point, sink at origin for compatibility
            x, y = self.interior_points[j]
            sources = [((x, y), 1.0), ((0.0, 0.0), -1.0)]
            
            u = solve_poisson_scipy(self.nodes, self.elements, sources, method)
            self.G[:, j] = u[self.boundary_indices]
        
        # Zero-mean columns
        self.G = self.G - np.mean(self.G, axis=0, keepdims=True)
        print("Done.")
    
    def solve_l2(self, u_measured: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
        """Solve with Tikhonov (L2) regularization."""
        if self.G is None:
            self.build_greens_matrix()
        
        u = u_measured - np.mean(u_measured)
        q = np.linalg.solve(self.G.T @ self.G + alpha * np.eye(self.n_interior), self.G.T @ u)
        return q - np.mean(q)
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, 
                 max_iter: int = 50) -> np.ndarray:
        """Solve with L1 (sparsity) regularization via IRLS."""
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
    
    def solve_tv(self, u_measured: np.ndarray, alpha: float = 1e-4, 
                 rho: float = 1.0, max_iter: int = 100) -> np.ndarray:
        """Solve with Total Variation regularization via ADMM."""
        if self.G is None:
            self.build_greens_matrix()
        
        from scipy.spatial import Delaunay
        
        u = u_measured - np.mean(u_measured)
        
        # Build gradient operator on interior mesh
        tri = Delaunay(self.interior_points)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                edges.add(tuple(sorted([s[i], s[(i+1)%3]])))
        
        D = np.zeros((len(edges), self.n_interior))
        for k, (i, j) in enumerate(edges):
            D[k, i] = 1
            D[k, j] = -1
        
        # ADMM
        q = np.zeros(self.n_interior)
        z = np.zeros(len(edges))
        w = np.zeros(len(edges))
        
        A_inv = np.linalg.inv(self.G.T @ self.G + rho * D.T @ D)
        Gtu = self.G.T @ u
        
        for _ in range(max_iter):
            q = A_inv @ (Gtu + rho * D.T @ (z - w))
            Dq = D @ q
            z = np.sign(Dq + w) * np.maximum(np.abs(Dq + w) - alpha/rho, 0)
            w = w + Dq - z
        
        return q - np.mean(q)
    
    def get_interior_positions(self) -> np.ndarray:
        """Return positions of interior nodes."""
        return self.interior_points.copy()


# =============================================================================
# FEM NONLINEAR INVERSE SOLVER (Continuous Source Positions)
# =============================================================================

class FEMNonlinearInverseSolver:
    """
    Nonlinear inverse solver with truly continuous source positions.
    
    Optimizes both source positions (ξ) and intensities (q) directly.
    Uses FEM forward solver internally.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    n_radial : int
        Mesh radial divisions
    n_angular : int
        Mesh angular divisions
    """
    
    def __init__(self, n_sources: int, n_radial: int = 15, n_angular: int = 30):
        self.n_sources = n_sources
        self.forward = FEMForwardSolver(n_radial, n_angular, use_dolfinx=False)
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the measured boundary data."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params) -> List[Tuple[Tuple[float, float], float]]:
        """
        Convert optimization parameters to source list.
        
        Parameters: [x1, y1, q1, x2, y2, q2, ..., xN, yN]
        Last intensity is computed from constraint: Σqₖ = 0
        """
        sources = []
        for i in range(self.n_sources - 1):
            x = params[3*i]
            y = params[3*i + 1]
            q = params[3*i + 2]
            sources.append(((x, y), q))
        
        # Last source: position from params, intensity from constraint
        x_last = params[3*(self.n_sources - 1)]
        y_last = params[3*(self.n_sources - 1) + 1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def _objective(self, params) -> float:
        """
        Objective function: ||u_computed - u_measured||²
        
        Includes penalty for sources outside domain.
        """
        sources = self._params_to_sources(params)
        
        # Penalty for sources outside domain
        for (x, y), _ in sources:
            if x**2 + y**2 >= 0.85**2:
                return 1e10
        
        # Forward solve
        u_computed = self.forward.solve(sources, method='interpolate')
        
        # Interpolate to match measurement points if needed
        if len(u_computed) != len(self.u_measured):
            # Interpolate u_computed to measurement angles
            from scipy.interpolate import interp1d
            interp = interp1d(self.forward.theta, u_computed, kind='linear', 
                            fill_value='extrapolate')
            # Assume u_measured corresponds to uniform theta
            theta_meas = np.linspace(0, 2*np.pi, len(self.u_measured), endpoint=False)
            u_computed = interp(theta_meas)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        self.history.append(misfit)
        
        return misfit
    
    def solve(self, method: str = 'L-BFGS-B', maxiter: int = 200) -> InverseResult:
        """
        Solve the nonlinear inverse problem.
        
        Parameters
        ----------
        method : str
            Optimization method: 'L-BFGS-B', 'differential_evolution', 
            'SLSQP', 'trust-constr', 'basinhopping'
        maxiter : int
            Maximum iterations
            
        Returns
        -------
        result : InverseResult
            Contains recovered sources and optimization info
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data first")
        
        self.history = []
        n = self.n_sources
        
        # Set up bounds: positions in disk, intensities unconstrained
        bounds = []
        for i in range(n):
            bounds.extend([(-0.8, 0.8), (-0.8, 0.8)])  # x, y
            if i < n - 1:
                bounds.append((-5.0, 5.0))  # q (except last)
        
        # Initial guess: sources distributed around circle
        x0 = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x0.extend([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
            if i < n - 1:
                x0.append(1.0 if i % 2 == 0 else -1.0)
        
        # Optimize
        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective, bounds, 
                maxiter=maxiter, seed=42, polish=True
            )
        elif method == 'basinhopping':
            from scipy.optimize import basinhopping
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            result = basinhopping(
                self._objective, x0, 
                minimizer_kwargs=minimizer_kwargs,
                niter=maxiter, seed=42
            )
        else:
            result = minimize(
                self._objective, x0, 
                method=method, bounds=bounds,
                options={'maxiter': maxiter}
            )
        
        # Extract sources
        sources = [Source(x, y, q) for (x, y), q in self._params_to_sources(result.x)]
        
        return InverseResult(
            sources=sources,
            residual=np.sqrt(result.fun),
            success=result.success if hasattr(result, 'success') else True,
            message=str(result.message) if hasattr(result, 'message') else '',
            iterations=result.nit if hasattr(result, 'nit') else len(self.history),
            history=self.history
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_synthetic_data_fem(sources, n_radial: int = 15, n_angular: int = 30,
                                noise_level: float = 0.0, seed: int = None):
    """
    Generate synthetic boundary measurements using FEM.
    
    Parameters
    ----------
    sources : list of ((x, y), q) tuples
    n_radial, n_angular : int
        Mesh resolution
    noise_level : float
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed
        
    Returns
    -------
    theta : array
        Boundary angles
    u : array
        Boundary values (with noise)
    """
    if seed is not None:
        np.random.seed(seed)
    
    forward = FEMForwardSolver(n_radial, n_angular)
    u = forward.solve(sources)
    
    if noise_level > 0:
        u += np.random.normal(0, noise_level, len(u))
    
    return forward.theta, u


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Keep old function names for compatibility
create_disk_mesh = create_disk_mesh_scipy
solve_poisson_zero_neumann = solve_poisson_scipy


if __name__ == "__main__":
    # Demo
    print("FEM Solver Demo")
    print("=" * 50)
    
    # Test sources
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.2), 1.0),
        ((-0.2, -0.3), -1.0),
        ((0.3, -0.4), -1.0),
    ]
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
    
    # Forward solve
    print("\n1. Forward solve (FEM)...")
    forward = FEMForwardSolver(n_radial=15, n_angular=30)
    u_measured = forward.solve(sources_true)
    print(f"   Computed {len(u_measured)} boundary values")
    
    # Add noise
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    # Linear inverse (L1)
    print("\n2. Linear inverse (L1 regularization)...")
    linear = FEMLinearInverseSolver(n_radial=10, n_angular=20)
    q_recovered = linear.solve_l1(u_measured, alpha=1e-3)
    
    # Find significant sources
    threshold = 0.1 * np.max(np.abs(q_recovered))
    significant = np.where(np.abs(q_recovered) > threshold)[0]
    print(f"   Found {len(significant)} significant sources")
    
    # Nonlinear inverse
    print("\n3. Nonlinear inverse (continuous positions)...")
    nonlinear = FEMNonlinearInverseSolver(n_sources=4, n_radial=15, n_angular=30)
    nonlinear.set_measured_data(u_measured)
    result = nonlinear.solve(method='L-BFGS-B', maxiter=100)
    
    print("\n   Recovered sources:")
    for i, s in enumerate(result.sources):
        print(f"     {i+1}: ({s.x:+.3f}, {s.y:+.3f}), q = {s.intensity:+.3f}")
    
    print(f"\n   Residual: {result.residual:.6e}")
    print(f"   Iterations: {result.iterations}")
    
    print("\nDemo complete!")
