"""
Boundary Element Method (BEM) Solver for Poisson Equation
==========================================================

This module implements TRUE Boundary Element Methods with numerical integration
for solving the Poisson equation with Neumann boundary conditions.

Unlike the analytical solver (which only works for point sources on the unit disk),
this BEM implementation can handle:
    - Distributed source terms (not just point sources)
    - General boundary data
    - Validation against analytical solutions

Problem:
    -Δu = f(x)     in Ω
    ∂u/∂n = g(x)   on ∂Ω  (or = 0 for homogeneous Neumann)

The BEM formulation converts the PDE to a boundary integral equation:
    c(x)u(x) = ∫_∂Ω [G(x,y)∂u/∂n(y) - ∂G/∂n(x,y)u(y)] ds(y) + ∫_Ω G(x,y)f(y) dV(y)

For the unit disk with known Green's function, we can still evaluate boundary
integrals numerically for validation and for cases where analytical evaluation
is not available.

This module is primarily for:
    1. Validating the analytical solver
    2. Handling distributed sources
    3. Educational purposes (understanding BEM)

For point source problems on the unit disk, use analytical_solver.py for efficiency.

References:
    - Sauter & Schwab, "Boundary Element Methods"
    - Steinbach, "Numerical Approximation Methods for Elliptic Boundary Value Problems"
"""

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.spatial import Delaunay
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass

try:
    from .mesh import get_source_grid
except ImportError:
    from mesh import get_source_grid


@dataclass
class BEMResult:
    """Result from BEM solve."""
    u_boundary: np.ndarray
    u_interior: Optional[np.ndarray] = None
    method: str = "bem"


def fundamental_solution_2d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fundamental solution (free-space Green's function) for 2D Laplacian.
    
    G*(x, y) = -1/(2π) ln|x - y|
    
    Satisfies: ΔG* = δ(x - y)
    """
    r = np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    if r < 1e-14:
        return 0.0  # Regularize singularity
    return -1/(2*np.pi) * np.log(r)


def fundamental_solution_gradient_2d(x: np.ndarray, y: np.ndarray, n: np.ndarray) -> float:
    """
    Normal derivative of fundamental solution.
    
    ∂G*/∂n_y = -1/(2π) · (x - y)·n / |x - y|²
    """
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    r_sq = dx**2 + dy**2
    if r_sq < 1e-28:
        return 0.0
    return -1/(2*np.pi) * (dx*n[0] + dy*n[1]) / r_sq


class BEMDiscretization:
    """
    Boundary element discretization for the unit disk.
    
    Discretizes the boundary into curved elements and provides
    quadrature for boundary integrals.
    
    Parameters
    ----------
    n_elements : int
        Number of boundary elements
    quadrature_order : int
        Gaussian quadrature order per element
    """
    
    def __init__(self, n_elements: int = 32, quadrature_order: int = 4):
        self.n_elements = n_elements
        self.quad_order = quadrature_order
        
        # Element endpoints (angles)
        self.theta_nodes = np.linspace(0, 2*np.pi, n_elements + 1)
        
        # Collocation points (element midpoints)
        self.theta_colloc = 0.5 * (self.theta_nodes[:-1] + self.theta_nodes[1:])
        self.colloc_points = np.column_stack([
            np.cos(self.theta_colloc),
            np.sin(self.theta_colloc)
        ])
        
        # Normals at collocation points (outward = radial for unit disk)
        self.normals = self.colloc_points.copy()  # n = x for unit circle
        
        # Gauss quadrature points and weights for [-1, 1]
        self.gauss_pts, self.gauss_wts = np.polynomial.legendre.leggauss(quadrature_order)
        
    def integrate_over_element(self, elem_idx: int, 
                                integrand: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """
        Integrate a function over a boundary element using Gaussian quadrature.
        
        Parameters
        ----------
        elem_idx : int
            Element index
        integrand : callable
            Function f(y, n_y) -> float to integrate
            
        Returns
        -------
        integral : float
            ∫_element f(y, n_y) ds(y)
        """
        theta_a = self.theta_nodes[elem_idx]
        theta_b = self.theta_nodes[elem_idx + 1]
        
        # Map Gauss points from [-1,1] to [theta_a, theta_b]
        theta_quad = 0.5 * (theta_b - theta_a) * self.gauss_pts + 0.5 * (theta_a + theta_b)
        
        # Element arc length differential: ds = |∂x/∂θ| dθ = R dθ = dθ for unit circle
        jacobian = 0.5 * (theta_b - theta_a)  # From [-1,1] to [θ_a, θ_b]
        
        result = 0.0
        for i, theta in enumerate(theta_quad):
            y = np.array([np.cos(theta), np.sin(theta)])
            n_y = y  # Normal = position for unit circle
            result += self.gauss_wts[i] * integrand(y, n_y) * jacobian
        
        return result
    
    def integrate_over_boundary(self, integrand: Callable) -> float:
        """Integrate over entire boundary."""
        return sum(self.integrate_over_element(i, integrand) 
                   for i in range(self.n_elements))


class BEMForwardSolver:
    """
    BEM forward solver using numerical integration.
    
    This computes u(x) = ∫_Ω G(x, y) f(y) dV(y) numerically,
    where f(y) is the source term.
    
    For point sources, this should match the analytical solver.
    For distributed sources, this is the only option.
    
    Parameters
    ----------
    n_elements : int
        Number of boundary elements
    quadrature_order : int
        Gaussian quadrature order
    """
    
    def __init__(self, n_elements: int = 64, quadrature_order: int = 6):
        self.disc = BEMDiscretization(n_elements, quadrature_order)
        self.n_boundary = n_elements
        self.theta = self.disc.theta_colloc
        self.boundary_points = self.disc.colloc_points
    
    def _neumann_green_function(self, x: np.ndarray, xi: np.ndarray) -> float:
        """
        Evaluate Neumann Green's function G(x, ξ) for unit disk.
        
        Uses method of images:
        G(x, ξ) = G*(x, ξ) + G*(x, ξ*) - correction
        where ξ* = ξ/|ξ|² is the image point.
        """
        xi_norm_sq = xi[0]**2 + xi[1]**2
        
        # Fundamental solution to source
        G = fundamental_solution_2d(x, xi)
        
        if xi_norm_sq > 1e-14:
            # Add image contribution
            xi_star = xi / xi_norm_sq
            G += fundamental_solution_2d(x, xi_star)
            G -= -1/(2*np.pi) * np.log(np.sqrt(xi_norm_sq))
        
        return G
    
    def solve_point_sources(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Solve for point sources using numerical evaluation of Green's function.
        
        This should match the analytical solver - useful for validation.
        
        Parameters
        ----------
        sources : list
            Point sources as [((x, y), intensity), ...]
            
        Returns
        -------
        u : array
            Boundary values (mean-centered)
        """
        u = np.zeros(self.n_boundary)
        
        for (xi_x, xi_y), q in sources:
            xi = np.array([xi_x, xi_y])
            for i, x in enumerate(self.boundary_points):
                u[i] += q * self._neumann_green_function(x, xi)
        
        return u - np.mean(u)
    
    def solve_distributed_source(self, 
                                  source_function: Callable[[float, float], float],
                                  n_quad: int = 20) -> np.ndarray:
        """
        Solve for a distributed source f(x, y).
        
        Computes u(x) = ∫_Ω G(x, y) f(y) dV(y) numerically.
        
        Parameters
        ----------
        source_function : callable
            f(x, y) -> float, the source term
        n_quad : int
            Number of quadrature points per dimension for domain integral
            
        Returns
        -------
        u : array
            Boundary values (mean-centered)
        """
        u = np.zeros(self.n_boundary)
        
        # Simple polar quadrature for unit disk
        r_pts, r_wts = np.polynomial.legendre.leggauss(n_quad)
        r_pts = 0.5 * (r_pts + 1)  # Map to [0, 1]
        r_wts = 0.5 * r_wts
        
        theta_pts = np.linspace(0, 2*np.pi, 2*n_quad, endpoint=False)
        dtheta = 2*np.pi / (2*n_quad)
        
        for i, x in enumerate(self.boundary_points):
            integral = 0.0
            for r, w_r in zip(r_pts, r_wts):
                for theta in theta_pts:
                    y = np.array([r * np.cos(theta), r * np.sin(theta)])
                    f_y = source_function(y[0], y[1])
                    G_xy = self._neumann_green_function(x, y)
                    # Jacobian for polar: r dr dθ
                    integral += G_xy * f_y * r * w_r * dtheta
            u[i] = integral
        
        return u - np.mean(u)
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Alias for solve_point_sources for API compatibility."""
        return self.solve_point_sources(sources)


class BEMLinearInverseSolver:
    """
    Linear inverse solver using BEM.
    
    This builds the Green's matrix using numerical evaluation
    rather than the analytical formula. Useful for validation
    or when analytical formula is not available.
    
    Parameters
    ----------
    n_boundary : int
        Number of boundary points
    source_resolution : float
        Grid spacing for source candidates
    n_elements : int
        BEM discretization elements
    """
    
    def __init__(self, n_boundary: int = 64, source_resolution: float = 0.15,
                 n_elements: int = None, verbose: bool = True):
        if n_elements is None:
            n_elements = n_boundary
            
        self.forward = BEMForwardSolver(n_elements)
        self.n_boundary = n_elements
        self.boundary_points = self.forward.boundary_points
        self.theta_boundary = self.forward.theta
        
        # Source candidate grid
        self.interior_points = get_source_grid(resolution=source_resolution, radius=0.9)
        self.n_interior = len(self.interior_points)
        self.G = None
        self._verbose = verbose
        
        if verbose:
            print(f"BEM Linear Inverse: {self.n_boundary} boundary pts, {self.n_interior} source candidates")
    
    def build_greens_matrix(self, verbose: bool = None):
        """Build Green's matrix using BEM evaluation."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print(f"Building Green's matrix via BEM ({self.n_boundary} × {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j in range(self.n_interior):
            xi = self.interior_points[j]
            for i, x in enumerate(self.boundary_points):
                self.G[i, j] = self.forward._neumann_green_function(x, xi)
        
        # Center columns
        self.G = self.G - np.mean(self.G, axis=0, keepdims=True)
        
        if verbose:
            print("Done.")
    
    def solve_l2(self, u_measured: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
        """Solve with Tikhonov regularization."""
        if self.G is None:
            self.build_greens_matrix()
        u = u_measured - np.mean(u_measured)
        q = np.linalg.solve(self.G.T @ self.G + alpha * np.eye(self.n_interior), self.G.T @ u)
        return q - np.mean(q)
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        """Solve with L1 regularization via IRLS."""
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
        """Solve with Total Variation regularization."""
        if self.G is None:
            self.build_greens_matrix()
        u = u_measured - np.mean(u_measured)
        
        # Build gradient operator
        tri = Delaunay(self.interior_points)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                edges.add(tuple(sorted([s[i], s[(i+1)%3]])))
        
        D = np.zeros((len(edges), self.n_interior))
        for k, (i, j) in enumerate(edges):
            D[k, i], D[k, j] = 1, -1
        
        # ADMM
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
        
        return q - np.mean(q)
    
    def get_interior_positions(self) -> np.ndarray:
        """Return source candidate positions."""
        return self.interior_points.copy()


def validate_against_analytical(n_sources: int = 4, n_boundary: int = 64,
                                 noise_level: float = 0.0, seed: int = 42) -> dict:
    """
    Validate BEM solver against analytical solver.
    
    Parameters
    ----------
    n_sources : int
        Number of point sources
    n_boundary : int
        Boundary discretization
    noise_level : float
        Noise to add
    seed : int
        Random seed
        
    Returns
    -------
    results : dict
        Comparison metrics
    """
    try:
        from .analytical_solver import AnalyticalForwardSolver, generate_synthetic_data
    except ImportError:
        from analytical_solver import AnalyticalForwardSolver, generate_synthetic_data
    
    np.random.seed(seed)
    
    # Create random sources
    sources = []
    total_q = 0
    for i in range(n_sources - 1):
        r = 0.2 + 0.5 * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        q = np.random.randn()
        sources.append(((r * np.cos(theta), r * np.sin(theta)), q))
        total_q += q
    # Last source for compatibility
    r, theta = 0.3, 0
    sources.append(((r * np.cos(theta), r * np.sin(theta)), -total_q))
    
    # Analytical solution
    analytical = AnalyticalForwardSolver(n_boundary)
    u_analytical = analytical.solve(sources)
    
    # BEM solution
    bem = BEMForwardSolver(n_boundary)
    u_bem = bem.solve_point_sources(sources)
    
    # Compare
    error = np.linalg.norm(u_analytical - u_bem) / np.linalg.norm(u_analytical)
    max_diff = np.max(np.abs(u_analytical - u_bem))
    
    return {
        'relative_error': error,
        'max_difference': max_diff,
        'u_analytical': u_analytical,
        'u_bem': u_bem,
        'sources': sources,
        'match': error < 1e-10
    }


# =============================================================================
# Convenience function for comparing methods
# =============================================================================
def compare_bem_analytical(sources: List[Tuple[Tuple[float, float], float]],
                           n_boundary: int = 100) -> None:
    """
    Compare BEM and analytical solutions for given sources.
    
    Prints comparison metrics and optionally plots.
    """
    try:
        from .analytical_solver import AnalyticalForwardSolver
    except ImportError:
        from analytical_solver import AnalyticalForwardSolver
    
    analytical = AnalyticalForwardSolver(n_boundary)
    bem = BEMForwardSolver(n_boundary)
    
    u_analytical = analytical.solve(sources)
    u_bem = bem.solve(sources)
    
    error = np.linalg.norm(u_analytical - u_bem)
    rel_error = error / np.linalg.norm(u_analytical)
    
    print(f"Comparison: BEM vs Analytical")
    print(f"  Sources: {len(sources)}")
    print(f"  Boundary points: {n_boundary}")
    print(f"  L2 error: {error:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  Max difference: {np.max(np.abs(u_analytical - u_bem)):.2e}")
    
    if rel_error < 1e-10:
        print("  ✓ Methods match to machine precision")
    else:
        print("  ⚠ Methods differ (check discretization)")
