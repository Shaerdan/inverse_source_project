"""
Finite Element Method Solver for Poisson Equation with Point Sources
=====================================================================

This module implements FEM-based forward and inverse solvers.

Formulations:
    - FEMForwardSolver: Forward problem with continuous source positions
    - FEMLinearInverseSolver: Linear inverse (sources on mesh grid)
    - FEMNonlinearInverseSolver: Nonlinear inverse (continuous source positions)

Mesh:
    - Uses shared mesh module for uniform triangular meshes (gmsh)
    - Forward mesh: finer, for FEM discretization
    - Source mesh: can be coarser, for candidate source locations
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import shared mesh utilities
try:
    from .mesh import (create_disk_mesh, get_source_grid, create_polygon_mesh, 
                       get_polygon_source_grid, create_ellipse_mesh, get_ellipse_source_grid,
                       get_ellipse_sensor_locations, get_polygon_sensor_locations)
except ImportError:
    from mesh import (create_disk_mesh, get_source_grid, create_polygon_mesh, 
                      get_polygon_source_grid, create_ellipse_mesh, get_ellipse_source_grid,
                      get_ellipse_sensor_locations, get_polygon_sensor_locations)


# Check for DOLFINx availability
try:
    import dolfinx
    HAS_DOLFINX = True
except ImportError:
    HAS_DOLFINX = False


# =============================================================================
# DATA CLASSES
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
# GEOMETRY UTILITIES
# =============================================================================

def find_containing_cell(nodes: np.ndarray, elements: np.ndarray, 
                         point: np.ndarray, delaunay: 'Delaunay' = None) -> Tuple[int, np.ndarray]:
    """
    Find the mesh cell containing a point and compute barycentric coordinates.
    
    If delaunay is provided, uses fast O(log N) lookup.
    Otherwise falls back to O(N) linear search.
    """
    x, y = point[0], point[1]
    
    # Fast path: use Delaunay if available
    if delaunay is not None:
        simplex_idx = delaunay.find_simplex(point.reshape(1, 2))[0]
        if simplex_idx >= 0:
            # Get triangle vertices
            cell = elements[simplex_idx]
            v0, v1, v2 = nodes[cell[0]], nodes[cell[1]], nodes[cell[2]]
            
            # Compute barycentric coordinates
            x1, y1 = v0
            x2, y2 = v1
            x3, y3 = v2
            
            det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
            if abs(det) > 1e-14:
                lam1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
                lam2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
                lam3 = 1 - lam1 - lam2
                return simplex_idx, np.array([lam1, lam2, lam3])
        return -1, np.array([0, 0, 0])
    
    # Slow fallback: linear search
    for cell_idx, cell in enumerate(elements):
        # Get triangle vertices
        v0 = nodes[cell[0]]
        v1 = nodes[cell[1]]
        v2 = nodes[cell[2]]
        
        # Quick bounding box check
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        
        if x < min_x - 1e-10 or x > max_x + 1e-10:
            continue
        if y < min_y - 1e-10 or y > max_y + 1e-10:
            continue
        
        # Compute barycentric coordinates
        x1, y1 = v0
        x2, y2 = v1
        x3, y3 = v2
        
        det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if abs(det) < 1e-14:
            continue
        
        lam1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / det
        lam2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / det
        lam3 = 1 - lam1 - lam2
        
        if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
            return cell_idx, np.array([lam1, lam2, lam3])
    
    return -1, np.array([0, 0, 0])


# =============================================================================
# FEM ASSEMBLY
# =============================================================================

def assemble_stiffness_matrix(nodes: np.ndarray, elements: np.ndarray) -> csr_matrix:
    """
    Assemble FEM stiffness matrix for P1 elements.
    """
    n_nodes = len(nodes)
    row, col, data = [], [], []
    
    for element in elements:
        coords = nodes[element]
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
        if area < 1e-14:
            continue
        
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        
        K_local = np.outer(b, b) + np.outer(c, c)
        K_local /= (4 * area)
        
        for i in range(3):
            for j in range(3):
                row.append(element[i])
                col.append(element[j])
                data.append(K_local[i, j])
    
    return csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))


def solve_poisson(nodes: np.ndarray, elements: np.ndarray,
                  sources: List[Tuple[Tuple[float, float], float]],
                  method: str = 'nullspace') -> np.ndarray:
    """
    Solve Poisson equation with point sources using FEM.
    
    -Δu = Σ qₖ δ(x - ξₖ)  in Ω
    ∂u/∂n = 0             on ∂Ω
    
    For Neumann problems, the solution is unique up to a constant.
    We enforce uniqueness via null space projection (like Firedrake/PETSc).
    
    Parameters
    ----------
    nodes : array, shape (n_nodes, 2)
    elements : array, shape (n_elements, 3)
    sources : list of ((x, y), intensity) tuples
    method : str
        'nullspace' - Project RHS and solution (default, recommended)
        'regularize' - Small diagonal perturbation only (legacy)
        'saddle_point' - Augmented system with GMRES (slow but exact)
    
    Returns
    -------
    u : array, shape (n_nodes,)
        Solution with zero mean
        
    Notes
    -----
    The 'nullspace' method mirrors Firedrake/PETSc's approach:
        nullspace = VectorSpaceBasis(constant=True)
        solve(a == L, nullspace=nullspace)
    
    It projects the RHS to be orthogonal to the null space (constants)
    before solving, then projects the solution after solving.
    """
    from scipy.sparse import bmat
    from scipy.sparse.linalg import gmres
    
    n_nodes = len(nodes)
    
    # Assemble stiffness matrix
    K = assemble_stiffness_matrix(nodes, elements)
    
    # Build RHS using barycentric interpolation
    f = np.zeros(n_nodes)
    
    for (xi_x, xi_y), q in sources:
        cell_idx, bary = find_containing_cell(nodes, elements, np.array([xi_x, xi_y]))
        
        if cell_idx < 0:
            # Fallback: snap to nearest node
            distances = np.sqrt((nodes[:, 0] - xi_x)**2 + (nodes[:, 1] - xi_y)**2)
            nearest = np.argmin(distances)
            f[nearest] += q
        else:
            cell_nodes = elements[cell_idx]
            for i, node in enumerate(cell_nodes):
                f[node] += q * bary[i]
    
    def project_nullspace(v):
        """Project to be orthogonal to constants."""
        return v - np.mean(v)
    
    if method == 'saddle_point':
        # =====================================================================
        # SADDLE-POINT with GMRES (exact but slower)
        # =====================================================================
        e = csr_matrix(np.ones((n_nodes, 1)))
        zero = csr_matrix((1, 1))
        A = bmat([[K, e], [e.T, zero]], format='csr')
        b = np.concatenate([f, [0.0]])
        x, info = gmres(A, b, atol=1e-12, rtol=1e-12)
        u = x[:-1]
        
    elif method == 'nullspace':
        # =====================================================================
        # NULL SPACE PROJECTION (like Firedrake/PETSc, recommended)
        # =====================================================================
        # 1. Project RHS to range(K) - ensures compatibility
        f = project_nullspace(f)
        
        # 2. Solve with small regularization for numerical stability
        eps = 1e-10
        K_reg = K + eps * diags([1.0] * n_nodes)
        u = spsolve(K_reg, f)
        
        # 3. Project solution - ensures uniqueness
        u = project_nullspace(u)
        
    else:  # 'regularize'
        # =====================================================================
        # REGULARIZATION ONLY (legacy)
        # =====================================================================
        eps = 1e-10
        K_reg = K + eps * diags([1.0] * n_nodes)
        u = spsolve(K_reg, f)
        u = project_nullspace(u)
    
    return u


# =============================================================================
# FEM FORWARD SOLVER
# =============================================================================

class FEMForwardSolver:
    """
    FEM Forward solver for Poisson equation with point sources.
    
    Uses uniform triangular mesh from shared mesh module.
    Sources at arbitrary positions via barycentric interpolation.
    
    Parameters
    ----------
    resolution : float
        Mesh element size (smaller = finer mesh). Default 0.1
    n_sensors : int
        Number of sensor locations on boundary. Default 100.
    sensor_locations : array, optional
        Custom sensor locations. If None, uses n_sensors evenly spaced.
    verbose : bool
        Print mesh info. Default True
    mesh_data : tuple, optional
        Custom mesh (nodes, elements, boundary_indices, interior_indices, sensor_indices).
        If provided, resolution is ignored.
    """
    
    def __init__(self, resolution: float = 0.1, n_sensors: int = 100,
                 sensor_locations: np.ndarray = None, verbose: bool = True,
                 mesh_data: Tuple = None):
        self.resolution = resolution
        
        # Determine sensor locations
        if sensor_locations is None:
            try:
                from .mesh import get_disk_sensor_locations
            except ImportError:
                from mesh import get_disk_sensor_locations
            sensor_locations = get_disk_sensor_locations(n_sensors, radius=1.0)
        
        self.sensor_locations = sensor_locations
        self.n_sensors = len(sensor_locations)
        
        # Use custom mesh if provided, otherwise create disk mesh with sensors
        if mesh_data is not None:
            if len(mesh_data) >= 5:
                self.nodes, self.elements, self.boundary_indices, self.interior_indices, self.sensor_indices = mesh_data
            else:
                self.nodes, self.elements, self.boundary_indices, self.interior_indices = mesh_data[:4]
                # Find sensor indices in the mesh
                self.sensor_indices = self._find_sensor_indices(sensor_locations)
        else:
            result = create_disk_mesh(resolution, sensor_locations=sensor_locations)
            self.nodes, self.elements, self.boundary_indices, self.interior_indices, self.sensor_indices = result
        
        # Store theta for reference (don't reorder sensor_indices!)
        sensor_points = self.nodes[self.sensor_indices]
        self.theta = np.arctan2(sensor_points[:, 1], sensor_points[:, 0])
        
        # Also sort boundary indices
        boundary_points = self.nodes[self.boundary_indices]
        boundary_theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        boundary_sort = np.argsort(boundary_theta)
        self.boundary_indices = self.boundary_indices[boundary_sort]
        self.n_boundary = len(self.boundary_indices)
        
        # Pre-compute element centroids and KDTree for fast cell lookup
        from scipy.spatial import cKDTree
        self._element_centroids = np.array([
            self.nodes[el].mean(axis=0) for el in self.elements
        ])
        self._centroid_tree = cKDTree(self._element_centroids)
        
        # Build node-to-elements adjacency for fast lookup
        self._node_to_elements = [[] for _ in range(len(self.nodes))]
        for el_idx, el in enumerate(self.elements):
            for node in el:
                self._node_to_elements[node].append(el_idx)
        
        # Pre-assemble and cache stiffness matrix (only depends on mesh!)
        self._K = None
        self._K_reg = None
        self._assemble_stiffness()
        
        if verbose:
            print(f"FEM mesh: {len(self.nodes)} nodes, {len(self.elements)} elements, "
                  f"{self.n_sensors} sensors")
    
    def _find_sensor_indices(self, sensor_locations: np.ndarray) -> np.ndarray:
        """Find mesh node indices closest to sensor locations."""
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((self.nodes[:, 0] - sx)**2 + (self.nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
        return sensor_indices
    
    def get_mesh_data(self) -> dict:
        """
        Export mesh data for saving.
        
        Returns
        -------
        dict with keys: nodes, elements, boundary_indices, interior_indices
        """
        return {
            'nodes': self.nodes.copy(),
            'elements': self.elements.copy(),
            'boundary_indices': self.boundary_indices.copy(),
            'interior_indices': self.interior_indices.copy() if hasattr(self, 'interior_indices') else np.array([]),
        }
    
    @classmethod
    def from_polygon(cls, vertices: List[Tuple[float, float]], resolution: float = 0.1,
                     verbose: bool = True) -> 'FEMForwardSolver':
        """
        Create solver for polygon domain.
        
        Parameters
        ----------
        vertices : list of (x, y)
            Polygon vertices in order (CCW)
        resolution : float
            Mesh element size
        """
        mesh_data = create_polygon_mesh(vertices, resolution)
        return cls(resolution=resolution, verbose=verbose, mesh_data=mesh_data)
    
    @classmethod
    def from_ellipse(cls, a: float, b: float, resolution: float = 0.1,
                     verbose: bool = True) -> 'FEMForwardSolver':
        """
        Create solver for ellipse domain.
        
        Parameters
        ----------
        a, b : float
            Semi-axes of ellipse
        resolution : float
            Mesh element size
        """
        mesh_data = create_ellipse_mesh(a, b, resolution)
        return cls(resolution=resolution, verbose=verbose, mesh_data=mesh_data)
    
    def _assemble_stiffness(self):
        """Assemble stiffness matrix for Neumann problem.
        
        For the Neumann problem -Δu = f with ∂u/∂n = 0, the solution is unique 
        only up to a constant. We handle this via small regularization:
        
            (K + εI)u = f
        
        For compatible data (∫f = 0, which holds when sources sum to zero),
        this produces the same unique zero-mean solution as:
        - Saddle-point with Lagrange multiplier (exact but unstable with direct solvers)
        - Projection onto range(K)
        
        The regularization shifts the null eigenvalue from 0 to ε, making the
        system invertible. With ε = 1e-10, the perturbation is negligible
        (verified: produces same accuracy as GMRES saddle-point).
        
        After solving, we center the solution (u = u - mean(u)) to ensure
        the zero-mean normalization.
        
        OPTIMIZATION: Pre-compute LU factorization for fast repeated solves.
        """
        from scipy.sparse.linalg import splu
        
        self._K = assemble_stiffness_matrix(self.nodes, self.elements)
        n_nodes = len(self.nodes)
        eps = 1e-10
        self._K_reg = self._K + eps * diags([1.0] * n_nodes)
        
        # Pre-compute LU factorization (MAJOR speedup for nonlinear inverse!)
        # splu requires CSC format
        self._K_lu = splu(self._K_reg.tocsc())
        
        # Store null space basis (normalized constant vector)
        self._nullspace = np.ones(n_nodes) / np.sqrt(n_nodes)
    
    def _project_nullspace(self, v: np.ndarray) -> np.ndarray:
        """Project vector to be orthogonal to null space (constants).
        
        This is equivalent to what Firedrake/PETSc does with:
            nullspace = VectorSpaceBasis(constant=True)
            
        For null space spanned by e = [1,1,...,1]/√n:
            v_projected = v - (v·e)e = v - mean(v)*ones
        """
        return v - np.mean(v)
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute solution at sensor locations for given sources.
        
        Uses null space projection (like Firedrake/PETSc):
        1. Project RHS to range(K): f̃ = f - mean(f)
        2. Solve (K + εI)u = f̃
        3. Project solution: ũ = u - mean(u)
        
        This ensures both compatibility and uniqueness.
        
        Parameters
        ----------
        sources : list of ((x, y), q) tuples
            Point sources (positions can be continuous)
            
        Returns
        -------
        u_sensors : array, shape (n_sensors,)
            Solution values at sensor locations (zero mean)
        """
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6e} ≠ 0")
        
        # Build RHS (this changes with source positions)
        f = self._build_rhs(sources)
        
        # Project RHS to be orthogonal to null space (ensures compatibility)
        f = self._project_nullspace(f)
        
        # Solve using pre-factored stiffness matrix (fast!)
        u = self._K_lu.solve(f)
        
        # Project solution to be orthogonal to null space (ensures uniqueness)
        u = self._project_nullspace(u)
        
        # Return solution at SENSOR locations (fixed measurement points)
        u_sensors = u[self.sensor_indices]
        return self._project_nullspace(u_sensors)
    
    def solve_at_boundary(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute solution at ALL boundary nodes (for backwards compatibility).
        
        Parameters
        ----------
        sources : list of ((x, y), q) tuples
            Point sources
            
        Returns
        -------
        u_boundary : array, shape (n_boundary,)
            Solution values at all boundary nodes
        """
        f = self._build_rhs(sources)
        f = self._project_nullspace(f)
        u = self._K_lu.solve(f)
        u = self._project_nullspace(u)
        return self._project_nullspace(u[self.boundary_indices])
    
    def _build_rhs(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Build RHS vector using barycentric interpolation with fast cell lookup."""
        n_nodes = len(self.nodes)
        f = np.zeros(n_nodes)
        
        for (xi_x, xi_y), q in sources:
            point = np.array([xi_x, xi_y])
            
            # Fast lookup: find nearest centroid, check that element + neighbors
            _, nearest_centroid_idx = self._centroid_tree.query(point, k=1)
            
            cell_idx = -1
            bary = None
            
            # Check the nearest element first
            candidates = [nearest_centroid_idx]
            # Also check elements sharing nodes with the nearest element
            for node in self.elements[nearest_centroid_idx]:
                candidates.extend(self._node_to_elements[node])
            
            for candidate_idx in candidates:
                cell = self.elements[candidate_idx]
                v0, v1, v2 = self.nodes[cell[0]], self.nodes[cell[1]], self.nodes[cell[2]]
                
                # Compute barycentric coordinates
                x1, y1 = v0
                x2, y2 = v1
                x3, y3 = v2
                
                det = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
                if abs(det) < 1e-14:
                    continue
                
                lam1 = ((y2 - y3)*(xi_x - x3) + (x3 - x2)*(xi_y - y3)) / det
                lam2 = ((y3 - y1)*(xi_x - x3) + (x1 - x3)*(xi_y - y3)) / det
                lam3 = 1 - lam1 - lam2
                
                if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
                    cell_idx = candidate_idx
                    bary = np.array([lam1, lam2, lam3])
                    break
            
            if cell_idx < 0:
                # Fallback: snap to nearest node
                distances = np.sqrt((self.nodes[:, 0] - xi_x)**2 + 
                                   (self.nodes[:, 1] - xi_y)**2)
                nearest = np.argmin(distances)
                f[nearest] += q
            else:
                cell_nodes = self.elements[cell_idx]
                for i, node in enumerate(cell_nodes):
                    f[node] += q * bary[i]
        
        return f
    
    def solve_full(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Return solution at all mesh nodes."""
        f = self._build_rhs(sources)
        f = self._project_nullspace(f)
        u = self._K_lu.solve(f)
        return self._project_nullspace(u)


# =============================================================================
# FEM LINEAR INVERSE SOLVER (Grid-Based / Distributed)
# =============================================================================

class FEMLinearInverseSolver:
    """
    Linear inverse solver using FEM Green's matrix.
    
    Sources constrained to interior mesh nodes (distributed formulation).
    Uses same mesh type as forward solver (uniform triangular).
    
    Parameters
    ----------
    forward_resolution : float
        Mesh resolution for FEM forward solves
    source_resolution : float
        Mesh resolution for source candidate grid (can be coarser)
    n_sensors : int
        Number of sensor/measurement locations on boundary. Default 100.
    verbose : bool
        Print info. Default True
    mesh_data : tuple, optional
        Custom mesh (nodes, elements, boundary_indices, interior_indices, sensor_indices)
    source_grid : array, optional
        Custom source candidate grid (N, 2) array
    sensor_locations : array, optional
        Custom sensor locations (M, 2) array. If None, uses n_sensors evenly spaced.
    """
    
    def __init__(self, forward_resolution: float = 0.1, source_resolution: float = 0.15,
                 n_sensors: int = 100, verbose: bool = True, 
                 mesh_data: Tuple = None, source_grid: np.ndarray = None,
                 sensor_locations: np.ndarray = None):
        
        # Determine sensor locations
        if sensor_locations is None:
            # Default: evenly spaced on unit circle
            try:
                from .mesh import get_disk_sensor_locations
            except ImportError:
                from mesh import get_disk_sensor_locations
            sensor_locations = get_disk_sensor_locations(n_sensors, radius=1.0)
        
        self.sensor_locations = sensor_locations
        self.n_sensors = len(sensor_locations)
        
        # Use custom mesh if provided (must include sensor points)
        if mesh_data is not None:
            if len(mesh_data) >= 5:
                self.nodes, self.elements, self.boundary_indices, interior_indices, self.sensor_indices = mesh_data
            else:
                # Old-style mesh_data without sensor indices - find them
                self.nodes, self.elements, self.boundary_indices, interior_indices = mesh_data[:4]
                self.sensor_indices = self._find_sensor_indices(sensor_locations)
        else:
            # Create mesh with sensor locations as required boundary points
            result = create_disk_mesh(forward_resolution,
                                      sensor_locations=sensor_locations)
            self.nodes, self.elements, self.boundary_indices, interior_indices, self.sensor_indices = result
        
        # Pre-assemble stiffness matrix for forward solves
        # Uses regularization (K + εI) for numerical stability with direct solvers.
        # Combined with null space projection, this is mathematically equivalent
        # to the saddle-point formulation used in Firedrake/PETSc.
        from scipy.sparse.linalg import splu
        
        self._K = assemble_stiffness_matrix(self.nodes, self.elements)
        n_nodes = len(self.nodes)
        self._K_reg = self._K + 1e-10 * diags([1.0] * n_nodes)
        
        # Pre-compute LU factorization for fast Green's matrix building
        self._K_lu = splu(self._K_reg.tocsc())
        
        # Store null space basis for projection (like Firedrake's VectorSpaceBasis)
        self._nullspace = np.ones(n_nodes) / np.sqrt(n_nodes)
        
        # Source candidate grid - use interior nodes from gmsh mesh (no margin filtering)
        if source_grid is not None:
            self.interior_points = source_grid
        else:
            # Use interior nodes of a gmsh mesh at source_resolution
            result = create_disk_mesh(source_resolution)
            source_nodes = result[0]
            source_interior = result[3]
            self.interior_points = source_nodes[source_interior]
        
        self.n_interior = len(self.interior_points)
        
        # Store theta for reference (don't reorder sensor_indices!)
        sensor_points = self.nodes[self.sensor_indices]
        self.theta = np.arctan2(sensor_points[:, 1], sensor_points[:, 0])
        
        # Also keep boundary indices for full boundary operations if needed
        boundary_points = self.nodes[self.boundary_indices]
        boundary_theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        boundary_sort = np.argsort(boundary_theta)
        self.boundary_indices = self.boundary_indices[boundary_sort]
        self.n_boundary = len(self.boundary_indices)
        
        self.G = None
        
        if verbose:
            print(f"FEM Linear: {self.n_sensors} sensors, {self.n_interior} source candidates")
    
    def _project_nullspace(self, v: np.ndarray) -> np.ndarray:
        """Project vector to be orthogonal to null space (constants).
        
        This mirrors Firedrake/PETSc's nullspace handling:
            nullspace = VectorSpaceBasis(constant=True)
        """
        return v - np.mean(v)
    
    def _find_sensor_indices(self, sensor_locations: np.ndarray) -> np.ndarray:
        """Find mesh node indices closest to sensor locations."""
        sensor_indices = np.zeros(len(sensor_locations), dtype=int)
        for i, (sx, sy) in enumerate(sensor_locations):
            dists = np.sqrt((self.nodes[:, 0] - sx)**2 + (self.nodes[:, 1] - sy)**2)
            sensor_indices[i] = np.argmin(dists)
        return sensor_indices
    
    def get_mesh_data(self) -> dict:
        """
        Export forward mesh data for saving.
        
        Returns
        -------
        dict with keys: nodes, elements, boundary_indices, interior_indices
        """
        return {
            'nodes': self.nodes.copy(),
            'elements': self.elements.copy(),
            'boundary_indices': self.boundary_indices.copy(),
            'interior_indices': np.array([i for i in range(len(self.nodes)) 
                                          if i not in self.boundary_indices]),
        }
    
    def get_source_grid(self) -> np.ndarray:
        """
        Export source grid (interior points) for saving.
        
        Returns
        -------
        np.ndarray : Source grid points (n_points, 2)
        """
        return self.interior_points.copy()
    
    @classmethod
    def from_polygon(cls, vertices: List[Tuple[float, float]], 
                     forward_resolution: float = 0.1, source_resolution: float = 0.15,
                     sensor_locations: np.ndarray = None, n_sensors: int = 100,
                     verbose: bool = True) -> 'FEMLinearInverseSolver':
        """
        Create solver for polygon domain.
        
        Parameters
        ----------
        vertices : list of (x, y)
            Polygon vertices in order (CCW)
        sensor_locations : array, optional
            Explicit sensor locations. If None, generates n_sensors on boundary.
        n_sensors : int
            Number of sensors if sensor_locations not provided
        """
        # Get or generate sensor locations
        if sensor_locations is None:
            sensor_locations = get_polygon_sensor_locations(vertices, n_sensors)
        
        # Create mesh WITH sensors embedded
        mesh_data = create_polygon_mesh(vertices, forward_resolution, sensor_locations=sensor_locations)
        source_grid = get_polygon_source_grid(vertices, source_resolution)
        return cls(forward_resolution=forward_resolution, source_resolution=source_resolution,
                   verbose=verbose, mesh_data=mesh_data, source_grid=source_grid,
                   sensor_locations=sensor_locations)
    
    @classmethod
    def from_ellipse(cls, a: float, b: float,
                     forward_resolution: float = 0.1, source_resolution: float = 0.15,
                     sensor_locations: np.ndarray = None, n_sensors: int = 100,
                     verbose: bool = True) -> 'FEMLinearInverseSolver':
        """
        Create solver for ellipse domain.
        
        Parameters
        ----------
        a, b : float
            Semi-axes of ellipse
        sensor_locations : array, optional
            Explicit sensor locations. If None, generates n_sensors on boundary.
        n_sensors : int
            Number of sensors if sensor_locations not provided
        """
        # Get or generate sensor locations
        if sensor_locations is None:
            sensor_locations = get_ellipse_sensor_locations(a, b, n_sensors)
        
        # Create mesh WITH sensors embedded
        mesh_data = create_ellipse_mesh(a, b, forward_resolution, sensor_locations=sensor_locations)
        source_grid = get_ellipse_source_grid(a, b, source_resolution)
        return cls(forward_resolution=forward_resolution, source_resolution=source_resolution,
                   verbose=verbose, mesh_data=mesh_data, source_grid=source_grid,
                   sensor_locations=sensor_locations)
    
    def build_greens_matrix(self, verbose: bool = True):
        """
        Build the Green's matrix by solving N forward problems.
        
        G[i, j] = u(sensor_i) when unit source at interior_point_j
        
        Uses null space projection (like Firedrake/PETSc):
        1. Project RHS to range(K)
        2. Solve (K + εI)u = f
        3. Project solution orthogonal to kernel
        
        This ensures the unique zero-mean solution for each column.
        """
        if verbose:
            print(f"Building FEM Green's matrix ({self.n_sensors} x {self.n_interior})...")
        
        self.G = np.zeros((self.n_sensors, self.n_interior))
        
        for j in range(self.n_interior):
            if verbose and j % 50 == 0:
                print(f"  Column {j}/{self.n_interior}")
            
            x, y = self.interior_points[j]
            # Unit source + sink at origin for compatibility
            sources = [((x, y), 1.0), ((0.0, 0.0), -1.0)]
            
            # Build RHS
            f = np.zeros(len(self.nodes))
            for (xi_x, xi_y), q in sources:
                cell_idx, bary = find_containing_cell(self.nodes, self.elements, 
                                                       np.array([xi_x, xi_y]))
                if cell_idx < 0:
                    distances = np.sqrt((self.nodes[:, 0] - xi_x)**2 + 
                                       (self.nodes[:, 1] - xi_y)**2)
                    nearest = np.argmin(distances)
                    f[nearest] += q
                else:
                    cell_nodes = self.elements[cell_idx]
                    for i, node in enumerate(cell_nodes):
                        f[node] += q * bary[i]
            
            # Project RHS to range(K) - ensures compatibility
            f = self._project_nullspace(f)
            
            # Solve with pre-factored stiffness matrix (fast!)
            u = self._K_lu.solve(f)
            
            # Project solution orthogonal to kernel - ensures uniqueness
            u = self._project_nullspace(u)
            
            # Extract solution at SENSOR locations (not all boundary nodes)
            self.G[:, j] = u[self.sensor_indices]
        
        # Final projection on columns (redundant but explicit)
        self.G = self.G - np.mean(self.G, axis=0, keepdims=True)
        if verbose:
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
                 method: str = 'admm', rho: float = 1.0, max_iter: int = 100,
                 verbose: bool = False) -> np.ndarray:
        """
        Solve with Total Variation regularization.
        
        Parameters
        ----------
        u_measured : array
            Boundary measurements
        alpha : float
            Regularization parameter
        method : str
            'admm' or 'chambolle_pock' (or 'cp')
        rho : float
            ADMM penalty parameter (for ADMM)
        max_iter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        q : array
            Recovered source intensities
        """
        if self.G is None:
            self.build_greens_matrix()
        
        from scipy.spatial import Delaunay
        
        u = u_measured - np.mean(u_measured)
        
        # Build gradient operator on source mesh
        tri = Delaunay(self.interior_points)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                edges.add(tuple(sorted([s[i], s[(i+1)%3]])))
        
        D = np.zeros((len(edges), self.n_interior))
        for k, (i, j) in enumerate(edges):
            D[k, i] = 1
            D[k, j] = -1
        
        if method.lower() in ('admm', 'tv_admm'):
            # ADMM implementation
            q = np.zeros(self.n_interior)
            z = np.zeros(len(edges))
            w = np.zeros(len(edges))
            
            A_inv = np.linalg.inv(self.G.T @ self.G + rho * D.T @ D)
            Gtu = self.G.T @ u
            
            for it in range(max_iter):
                q = A_inv @ (Gtu + rho * D.T @ (z - w))
                Dq = D @ q
                z = np.sign(Dq + w) * np.maximum(np.abs(Dq + w) - alpha/rho, 0)
                w = w + Dq - z
                
                if verbose and it % 20 == 0:
                    energy = 0.5 * np.linalg.norm(self.G @ q - u)**2 + alpha * np.sum(np.abs(D @ q))
                    print(f"  ADMM iter {it}: energy = {energy:.6e}")
        
        elif method.lower() in ('chambolle_pock', 'cp', 'tv_cp'):
            # Chambolle-Pock primal-dual
            from .regularization import solve_tv_chambolle_pock
            result = solve_tv_chambolle_pock(self.G, u, D, alpha=alpha,
                                             max_iter=max_iter, verbose=verbose)
            q = result.q
        
        else:
            raise ValueError(f"Unknown TV method: {method}. Use 'admm' or 'chambolle_pock'")
        
        return q - np.mean(q)
    
    def get_interior_positions(self) -> np.ndarray:
        """Return positions of source candidate grid."""
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
    resolution : float
        Mesh resolution for FEM forward solver
    verbose : bool
        Print info. Default False (quiet during optimization)
    mesh_data : tuple, optional
        Custom mesh (nodes, elements, boundary_indices, interior_indices)
    bounds : tuple, optional
        Custom bounds for source positions ((x_min, x_max), (y_min, y_max))
    domain_type : str, optional
        'disk', 'ellipse', 'polygon'. Default 'disk'.
    domain_params : dict, optional
        Domain parameters (e.g., vertices for polygon, a/b for ellipse)
    """
    
    def __init__(self, n_sources: int, resolution: float = 0.1, verbose: bool = False,
                 mesh_data: Tuple = None, bounds: Tuple = None,
                 domain_type: str = 'disk', domain_params: dict = None):
        self.n_sources = n_sources
        self.forward = FEMForwardSolver(resolution=resolution, verbose=verbose, mesh_data=mesh_data)
        self.u_measured = None
        self.history = []
        self.domain_type = domain_type
        self.domain_params = domain_params or {}
        
        # Default bounds for unit disk
        if bounds is None:
            self.x_bounds = (-0.9, 0.9)
            self.y_bounds = (-0.9, 0.9)
        else:
            self.x_bounds = bounds[0]
            self.y_bounds = bounds[1]
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, vertices: List[Tuple[float, float]]) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    @classmethod
    def from_polygon(cls, vertices: List[Tuple[float, float]], n_sources: int,
                     resolution: float = 0.1, n_sensors: int = 100,
                     verbose: bool = False, sensor_locations: np.ndarray = None,
                     mesh_data: Tuple = None) -> 'FEMNonlinearInverseSolver':
        """
        Create solver for polygon domain.
        
        Parameters
        ----------
        vertices : list of (x, y)
            Polygon vertices in order (CCW)
        n_sources : int
            Number of sources to recover
        resolution : float
            Mesh resolution
        n_sensors : int
            Number of boundary sensors
        sensor_locations : array, optional
            Explicit sensor locations. If provided, n_sensors is ignored.
        mesh_data : tuple, optional
            Pre-built mesh data to reuse.
        """
        vertices_arr = np.array(vertices)
        n_vertices = len(vertices)
        
        # Use provided sensor locations or generate them
        if sensor_locations is None:
            # Generate sensor locations evenly distributed on polygon boundary
            # Compute edge lengths
            edges = np.diff(np.vstack([vertices_arr, vertices_arr[0:1]]), axis=0)
            edge_lengths = np.linalg.norm(edges, axis=1)
            total_length = np.sum(edge_lengths)
            
            # Distribute sensors proportionally along edges
            sensor_locations = []
            cumulative = 0
            sensor_spacing = total_length / n_sensors
            
            for i in range(n_vertices):
                v_start = vertices_arr[i]
                v_end = vertices_arr[(i + 1) % n_vertices]
                edge_len = edge_lengths[i]
                
                # Add sensors along this edge
                t = 0
                while cumulative + t * edge_len < len(sensor_locations) * sensor_spacing + sensor_spacing:
                    t_next = ((len(sensor_locations) + 1) * sensor_spacing - cumulative) / edge_len
                    if t_next <= 1:
                        pt = v_start + t_next * (v_end - v_start)
                        sensor_locations.append(pt)
                    else:
                        break
                cumulative += edge_len
            
            sensor_locations = np.array(sensor_locations[:n_sensors])
        else:
            sensor_locations = np.asarray(sensor_locations)
        
        # Create or reuse mesh
        if mesh_data is None:
            mesh_data = create_polygon_mesh(vertices, resolution, sensor_locations=sensor_locations)
        
        # Compute bounds from vertices
        x_min, y_min = vertices_arr.min(axis=0)
        x_max, y_max = vertices_arr.max(axis=0)
        margin = 0.1 * min(x_max - x_min, y_max - y_min)
        bounds = ((x_min + margin, x_max - margin), (y_min + margin, y_max - margin))
        
        return cls(n_sources=n_sources, resolution=resolution, verbose=verbose,
                   mesh_data=mesh_data, bounds=bounds,
                   domain_type='polygon', domain_params={'vertices': vertices})
    
    @classmethod
    def from_ellipse(cls, a: float, b: float, n_sources: int,
                     resolution: float = 0.1, n_sensors: int = 100,
                     verbose: bool = False, sensor_locations: np.ndarray = None,
                     mesh_data: Tuple = None) -> 'FEMNonlinearInverseSolver':
        """
        Create solver for ellipse domain.
        
        Parameters
        ----------
        a, b : float
            Semi-axes of ellipse
        n_sources : int
            Number of sources to recover
        resolution : float
            Mesh resolution
        n_sensors : int
            Number of boundary sensors
        sensor_locations : array, optional
            Explicit sensor locations. If provided, n_sensors is ignored.
        mesh_data : tuple, optional
            Pre-built mesh data to reuse.
        """
        # Use provided sensor locations or generate them
        if sensor_locations is None:
            theta = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
            sensor_locations = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
        
        # Create mesh with sensors embedded or reuse provided mesh
        if mesh_data is None:
            mesh_data = create_ellipse_mesh(a, b, resolution, sensor_locations=sensor_locations)
        bounds = ((-0.85*a, 0.85*a), (-0.85*b, 0.85*b))
        return cls(n_sources=n_sources, resolution=resolution, verbose=verbose,
                   mesh_data=mesh_data, bounds=bounds,
                   domain_type='ellipse', domain_params={'a': a, 'b': b})
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the measured boundary data."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params) -> List[Tuple[Tuple[float, float], float]]:
        """Convert optimization parameters to source list."""
        sources = []
        for i in range(self.n_sources - 1):
            x = params[3*i]
            y = params[3*i + 1]
            q = params[3*i + 2]
            sources.append(((x, y), q))
        
        # Last source: intensity from constraint Σqₖ = 0
        x_last = params[3*(self.n_sources - 1)]
        y_last = params[3*(self.n_sources - 1) + 1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def _objective(self, params) -> float:
        """Objective function: ||u_computed - u_measured||²"""
        sources = self._params_to_sources(params)
        
        # Penalty for sources outside domain
        for (x, y), _ in sources:
            if self.domain_type == 'polygon':
                vertices = self.domain_params.get('vertices', [])
                if vertices and not self._point_in_polygon(x, y, vertices):
                    return 1e10
            elif self.domain_type == 'ellipse':
                a = self.domain_params.get('a', 1.0)
                b = self.domain_params.get('b', 1.0)
                if (x/a)**2 + (y/b)**2 >= 0.85**2:
                    return 1e10
            else:  # disk
                if x**2 + y**2 >= 0.85**2:
                    return 1e10
        
        u_computed = self.forward.solve(sources)
        
        # Interpolate if needed
        if len(u_computed) != len(self.u_measured):
            from scipy.interpolate import interp1d
            interp = interp1d(self.forward.theta, u_computed, kind='linear', 
                            fill_value='extrapolate')
            theta_meas = np.linspace(0, 2*np.pi, len(self.u_measured), endpoint=False)
            u_computed = interp(theta_meas)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        self.history.append(misfit)
        
        return misfit
    
    def solve(self, method: str = 'L-BFGS-B', maxiter: int = 200, 
              n_restarts: int = 1, init_from: str = 'circle') -> InverseResult:
        """
        Solve the nonlinear inverse problem.
        
        Parameters
        ----------
        method : str
            'L-BFGS-B', 'differential_evolution', 'SLSQP', 'basinhopping'
        maxiter : int
            Maximum iterations
        n_restarts : int
            Number of random restarts for local optimizers (best result kept)
        init_from : str
            'circle' - sources on circle (default)
            'random' - random positions
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data first")
        
        self.history = []
        n = self.n_sources
        
        # Bounds - use stored bounds for custom domains
        bounds = []
        for i in range(n):
            bounds.extend([self.x_bounds, self.y_bounds])
            if i < n - 1:
                bounds.append((-5.0, 5.0))
        
        best_result = None
        best_fun = np.inf
        
        # For global optimizers, just run once
        if method == 'differential_evolution':
            result = differential_evolution(self._objective, bounds, maxiter=maxiter, 
                                           seed=42, polish=True, workers=1)
            best_result = result
        elif method == 'basinhopping':
            from scipy.optimize import basinhopping
            x0 = self._get_initial_guess(init_from, 0)
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            result = basinhopping(self._objective, x0, minimizer_kwargs=minimizer_kwargs,
                                 niter=maxiter, seed=42)
            best_result = result
        else:
            # Local optimizer with restarts
            for restart in range(n_restarts):
                x0 = self._get_initial_guess(init_from, restart)
                result = minimize(self._objective, x0, method=method, bounds=bounds,
                                options={'maxiter': maxiter})
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
        
        sources = [Source(x, y, q) for (x, y), q in self._params_to_sources(best_result.x)]
        
        return InverseResult(
            sources=sources,
            residual=np.sqrt(best_result.fun),
            success=best_result.success if hasattr(best_result, 'success') else True,
            message=str(best_result.message) if hasattr(best_result, 'message') else '',
            iterations=best_result.nit if hasattr(best_result, 'nit') else len(self.history),
            history=self.history
        )
    
    def _get_initial_guess(self, init_from: str, seed: int) -> list:
        """Generate initial guess for optimization.
        
        For non-disk domains, uses interior mesh points or domain centroid
        to ensure initial guesses are valid.
        """
        n = self.n_sources
        x0 = []
        
        # Get domain bounds
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        
        # Compute domain centroid and safe radius
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        
        # Safe radius is half the smaller dimension, shrunk by 0.6
        safe_r = min(x_max - cx, y_max - cy) * 0.6
        
        if init_from == 'random' or seed > 0:
            np.random.seed(42 + seed)
            for i in range(n):
                # Use random angle and safe radius relative to domain centroid
                r = safe_r * (0.3 + 0.5 * np.random.rand())  # r in [0.3*safe_r, 0.8*safe_r]
                angle = 2 * np.pi * np.random.rand()
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                
                # Clamp to bounds
                x = np.clip(x, x_min * 0.9, x_max * 0.9)
                y = np.clip(y, y_min * 0.9, y_max * 0.9)
                
                x0.extend([x, y])
                if i < n - 1:
                    x0.append(np.random.randn())
        else:
            # Circle initialization around centroid
            for i in range(n):
                angle = 2 * np.pi * i / n
                x = cx + safe_r * 0.5 * np.cos(angle)
                y = cy + safe_r * 0.5 * np.sin(angle)
                x0.extend([x, y])
                if i < n - 1:
                    x0.append(1.0 if i % 2 == 0 else -1.0)
        
        return x0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_synthetic_data_fem(sources, resolution: float = 0.1,
                                noise_level: float = 0.0, seed: int = None):
    """Generate synthetic boundary measurements using FEM."""
    if seed is not None:
        np.random.seed(seed)
    
    forward = FEMForwardSolver(resolution=resolution, verbose=False)
    u = forward.solve(sources)
    
    if noise_level > 0:
        u += np.random.normal(0, noise_level, len(u))
    
    return forward.theta, u


# Backward compatibility aliases
def create_disk_mesh_scipy(resolution=0.1, radius=1.0):
    """Alias for backward compatibility."""
    return create_disk_mesh(resolution, radius)

def solve_poisson_scipy(nodes, elements, sources, method='interpolate'):
    """Alias for backward compatibility."""
    return solve_poisson(nodes, elements, sources)


if __name__ == "__main__":
    print("FEM Solver Demo")
    print("=" * 50)
    
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.2), 1.0),
        ((-0.2, -0.3), -1.0),
        ((0.3, -0.4), -1.0),
    ]
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
    
    print("\n1. Forward solve...")
    forward = FEMForwardSolver(resolution=0.1)
    u_measured = forward.solve(sources_true)
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    print("\n2. Linear inverse (L1)...")
    linear = FEMLinearInverseSolver(forward_resolution=0.1, source_resolution=0.15)
    q_recovered = linear.solve_l1(u_measured, alpha=1e-3)
    
    threshold = 0.1 * np.max(np.abs(q_recovered))
    significant = np.where(np.abs(q_recovered) > threshold)[0]
    print(f"   Found {len(significant)} significant sources")
    
    print("\n3. Nonlinear inverse...")
    nonlinear = FEMNonlinearInverseSolver(n_sources=4, resolution=0.1)
    nonlinear.set_measured_data(u_measured)
    result = nonlinear.solve(method='L-BFGS-B', maxiter=100)
    
    print("\n   Recovered sources:")
    for i, s in enumerate(result.sources):
        print(f"     {i+1}: ({s.x:+.3f}, {s.y:+.3f}), q = {s.intensity:+.3f}")
    
    print(f"\n   Residual: {result.residual:.6e}")
    print("\nDemo complete!")
