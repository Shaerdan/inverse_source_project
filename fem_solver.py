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
from .mesh import create_disk_mesh, get_source_grid


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
                         point: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Find the mesh cell containing a point and compute barycentric coordinates.
    Uses bounding box pre-check for speed.
    """
    x, y = point[0], point[1]
    
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
                  sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
    """
    Solve Poisson equation with point sources using FEM.
    
    -Δu = Σ qₖ δ(x - ξₖ)  in Ω
    ∂u/∂n = 0             on ∂Ω
    """
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
    
    # Regularize for Neumann problem
    eps = 1e-10
    K_reg = K + eps * diags([1.0] * n_nodes)
    
    # Solve
    u = spsolve(K_reg, f)
    u = u - np.mean(u)  # Zero mean
    
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
    verbose : bool
        Print mesh info. Default True
    """
    
    def __init__(self, resolution: float = 0.1, verbose: bool = True):
        self.resolution = resolution
        
        # Create mesh using shared module
        self.nodes, self.elements, self.boundary_indices, self.interior_indices = \
            create_disk_mesh(resolution)
        
        # Boundary info (sorted by angle for interpolation)
        boundary_points = self.nodes[self.boundary_indices]
        self.theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        sort_idx = np.argsort(self.theta)
        self.theta = self.theta[sort_idx]
        self.boundary_indices = self.boundary_indices[sort_idx]
        self.n_boundary = len(self.boundary_indices)
        
        # Pre-assemble and cache stiffness matrix (only depends on mesh!)
        self._K = None
        self._K_reg = None
        self._assemble_stiffness()
        
        if verbose:
            print(f"FEM mesh: {len(self.nodes)} nodes, {len(self.elements)} elements, "
                  f"{self.n_boundary} boundary points")
    
    def _assemble_stiffness(self):
        """Assemble stiffness matrix once (only depends on mesh geometry)."""
        self._K = assemble_stiffness_matrix(self.nodes, self.elements)
        # Pre-compute regularized version for Neumann problem
        n_nodes = len(self.nodes)
        eps = 1e-10
        self._K_reg = self._K + eps * diags([1.0] * n_nodes)
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute boundary values for given sources.
        
        Parameters
        ----------
        sources : list of ((x, y), q) tuples
            Point sources (positions can be continuous)
            
        Returns
        -------
        u_boundary : array
            Solution values at boundary points
        """
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6e} ≠ 0")
        
        # Build RHS (this changes with source positions)
        f = self._build_rhs(sources)
        
        # Solve using cached stiffness matrix
        u = spsolve(self._K_reg, f)
        u = u - np.mean(u)
        
        return u[self.boundary_indices] - np.mean(u[self.boundary_indices])
    
    def _build_rhs(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Build RHS vector using barycentric interpolation."""
        n_nodes = len(self.nodes)
        f = np.zeros(n_nodes)
        
        for (xi_x, xi_y), q in sources:
            cell_idx, bary = find_containing_cell(self.nodes, self.elements, 
                                                   np.array([xi_x, xi_y]))
            
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
        u = spsolve(self._K_reg, f)
        return u - np.mean(u)


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
    verbose : bool
        Print info. Default True
    """
    
    def __init__(self, forward_resolution: float = 0.1, source_resolution: float = 0.15,
                 verbose: bool = True):
        # Forward mesh (finer)
        self.nodes, self.elements, self.boundary_indices, _ = \
            create_disk_mesh(forward_resolution)
        
        # Pre-assemble stiffness matrix for forward solves
        self._K = assemble_stiffness_matrix(self.nodes, self.elements)
        n_nodes = len(self.nodes)
        self._K_reg = self._K + 1e-10 * diags([1.0] * n_nodes)
        
        # Source candidate mesh (can be coarser)
        source_nodes, _, _, source_interior = create_disk_mesh(source_resolution)
        
        # Filter to r < 0.9 to stay well inside domain
        self.interior_points = source_nodes[source_interior]
        radii = np.sqrt(self.interior_points[:, 0]**2 + self.interior_points[:, 1]**2)
        mask = radii < 0.9
        self.interior_points = self.interior_points[mask]
        self.n_interior = len(self.interior_points)
        
        # Sort boundary by angle
        boundary_points = self.nodes[self.boundary_indices]
        theta = np.arctan2(boundary_points[:, 1], boundary_points[:, 0])
        sort_idx = np.argsort(theta)
        self.boundary_indices = self.boundary_indices[sort_idx]
        self.theta = theta[sort_idx]
        self.n_boundary = len(self.boundary_indices)
        
        self.G = None
        
        if verbose:
            print(f"FEM Linear: {self.n_boundary} boundary, {self.n_interior} source candidates")
    
    def build_greens_matrix(self, verbose: bool = True):
        """
        Build the Green's matrix by solving N forward problems.
        
        G[i, j] = u(boundary_point_i) when unit source at interior_point_j
        
        Uses cached stiffness matrix for speed.
        """
        if verbose:
            print(f"Building FEM Green's matrix ({self.n_boundary} x {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
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
            
            # Solve with cached matrix
            u = spsolve(self._K_reg, f)
            u = u - np.mean(u)
            self.G[:, j] = u[self.boundary_indices]
        
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
    """
    
    def __init__(self, n_sources: int, resolution: float = 0.1, verbose: bool = False):
        self.n_sources = n_sources
        self.forward = FEMForwardSolver(resolution=resolution, verbose=verbose)
        self.u_measured = None
        self.history = []
    
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
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.extend([(-0.8, 0.8), (-0.8, 0.8)])
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
        """Generate initial guess for optimization."""
        n = self.n_sources
        x0 = []
        
        if init_from == 'random' or seed > 0:
            np.random.seed(42 + seed)
            for i in range(n):
                r = 0.3 + 0.4 * np.random.rand()
                angle = 2 * np.pi * np.random.rand()
                x0.extend([r * np.cos(angle), r * np.sin(angle)])
                if i < n - 1:
                    x0.append(np.random.randn())
        else:
            # Circle initialization
            for i in range(n):
                angle = 2 * np.pi * i / n
                x0.extend([0.4 * np.cos(angle), 0.4 * np.sin(angle)])
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
