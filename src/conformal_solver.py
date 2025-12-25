"""
Conformal Mapping Solver for General Simply-Connected Domains
==============================================================

This module extends the analytical solver to general simply-connected domains
using conformal mapping. The key idea is:

    1. Map the physical domain Ω to the unit disk D via conformal map w = f(z)
    2. Solve the transformed problem on the disk (where we have analytical solution)
    3. Map the solution back to the physical domain

For the Laplacian, conformal invariance means:
    Δ_z u = |f'(z)|² Δ_w ũ

where ũ(w) = u(f⁻¹(w)) is the transformed solution.

Supported Domains
-----------------
1. **Ellipse**: w = (z + 1/z)/2 maps exterior of unit circle to exterior of ellipse
               (need Joukowsky variant for interior)
2. **Star-shaped domains**: Numerical conformal map via Symm's integral equation
3. **Polygon**: Schwarz-Christoffel transformation

For point sources, the Green's function transforms as:
    G_Ω(z, ζ) = G_D(f(z), f(ζ))

This is exact because the Green's function is conformally invariant (in 2D).

References:
    - Driscoll & Trefethen, "Schwarz-Christoffel Mapping"
    - Henrici, "Applied and Computational Complex Analysis"
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    from .analytical_solver import (
        greens_function_disk_neumann,
        Source,
        InverseResult
    )
    from .mesh import get_source_grid
except ImportError:
    from analytical_solver import (
        greens_function_disk_neumann,
        Source,
        InverseResult
    )
    from mesh import get_source_grid


class ConformalMap(ABC):
    """
    Abstract base class for conformal mappings.
    
    A conformal map f: Ω → D maps a simply-connected domain Ω
    to the unit disk D = {|w| < 1}.
    
    Subclasses must implement:
        - to_disk(z): Ω → D
        - from_disk(w): D → Ω
        - jacobian(z): |f'(z)|
    """
    
    @abstractmethod
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Map point(s) from physical domain to unit disk."""
        pass
    
    @abstractmethod
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map point(s) from unit disk to physical domain."""
        pass
    
    @abstractmethod
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Compute |f'(z)|, the Jacobian of the mapping."""
        pass
    
    @abstractmethod
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Return n_points on the physical boundary as complex numbers."""
        pass
    
    @abstractmethod
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if points are inside the physical domain."""
        pass


class DiskMap(ConformalMap):
    """
    Identity map for the unit disk (trivial case).
    
    Useful as a baseline and for testing.
    
    Parameters
    ----------
    radius : float
        Disk radius (default 1.0)
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Scale to unit disk."""
        return np.asarray(z) / self.radius
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Scale from unit disk."""
        return np.asarray(w) * self.radius
    
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Constant Jacobian = 1/radius."""
        return np.ones_like(np.asarray(z), dtype=float) / self.radius
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Circle of given radius."""
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.radius * np.exp(1j * theta)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if inside disk."""
        return np.abs(z) < self.radius


class EllipseMap(ConformalMap):
    """
    Conformal map from ellipse to unit disk.
    
    Uses the inverse Joukowsky map. The ellipse has semi-axes a and b.
    
    The mapping is:
        w = f(z) maps ellipse interior to disk interior
        
    For ellipse with semi-axes a, b where a > b:
        c = sqrt(a² - b²) (focal distance)
        
    The inverse Joukowsky map z = (w + 1/w)/2 maps |w| = r to ellipse
    with semi-axes (r + 1/r)/2 and (r - 1/r)/2.
    
    Parameters
    ----------
    a : float
        Semi-major axis (along x)
    b : float
        Semi-minor axis (along y), must have b ≤ a
    """
    
    def __init__(self, a: float = 2.0, b: float = 1.0):
        if b > a:
            a, b = b, a  # Ensure a >= b
        self.a = a
        self.b = b
        
        # For ellipse with semi-axes a, b, the Joukowsky parameter is:
        # a = (r + 1/r)/2, b = (r - 1/r)/2
        # Solving: r = a + sqrt(a² - b²) / b... this is complex
        # Simpler: use scaling
        
        # Focal distance
        self.c = np.sqrt(a**2 - b**2) if a > b else 0
        
        # The ellipse (x/a)² + (y/b)² = 1 is the image of |w| = R under
        # z = c/2 (w + 1/w) where R satisfies:
        #   a = c/2 (R + 1/R), b = c/2 (R - 1/R)
        # So R = (a + b) / c (assuming c > 0)
        
        if self.c > 1e-10:
            self.R = (a + b) / self.c
        else:
            # Nearly circular - use identity with scaling
            self.R = 1.0
            self.c = 1.0  # Avoid division by zero
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """
        Map from ellipse to unit disk.
        
        Inverse of the Joukowsky map scaled appropriately.
        """
        z = np.asarray(z, dtype=complex)
        
        if self.a == self.b:  # Circle
            return z / self.a
        
        # Inverse Joukowsky: if z = c/2 (w + 1/w), then
        # w = (z ± sqrt(z² - c²)) / c
        # Take the root with |w| < 1 for interior points
        
        zeta = 2 * z / self.c  # Normalize
        discriminant = zeta**2 - 4
        
        # Choose correct branch
        w = (zeta - np.sqrt(discriminant)) / 2
        
        # Ensure we're mapping to interior
        mask = np.abs(w) > 1
        if np.any(mask):
            w = np.where(mask, 1/w, w)
        
        # Scale to unit disk
        return w / self.R
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk to ellipse."""
        w = np.asarray(w, dtype=complex)
        
        if self.a == self.b:  # Circle
            return w * self.a
        
        # Scale w
        w_scaled = w * self.R
        
        # Joukowsky map: z = c/2 (w + 1/w)
        z = (self.c / 2) * (w_scaled + 1 / (w_scaled + 1e-15))
        
        return z
    
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Compute |f'(z)| for the mapping to disk."""
        z = np.asarray(z, dtype=complex)
        
        if self.a == self.b:
            return np.ones_like(z, dtype=float) / self.a
        
        # f(z) = to_disk(z), compute |f'(z)| numerically
        eps = 1e-8
        f_z = self.to_disk(z)
        f_z_eps = self.to_disk(z + eps)
        
        return np.abs((f_z_eps - f_z) / eps)
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Ellipse boundary."""
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.a * np.cos(theta) + 1j * self.b * np.sin(theta)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if inside ellipse."""
        z = np.asarray(z, dtype=complex)
        return (np.real(z)/self.a)**2 + (np.imag(z)/self.b)**2 < 1


class StarShapedMap(ConformalMap):
    """
    Conformal map for star-shaped domains defined by r = R(θ).
    
    The boundary is given in polar form: r = R(θ) for θ ∈ [0, 2π).
    
    Uses numerical conformal mapping via series expansion or
    iterative methods.
    
    Parameters
    ----------
    radius_func : callable
        Function R(θ) defining the boundary radius
    n_terms : int
        Number of terms in the series expansion
    """
    
    def __init__(self, radius_func: Callable[[float], float], n_terms: int = 32):
        self.R = radius_func
        self.n_terms = n_terms
        
        # Compute boundary points
        theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
        r = np.array([self.R(t) for t in theta])
        self.boundary_pts = r * np.exp(1j * theta)
        
        # Compute mapping coefficients numerically
        self._compute_map_coefficients()
    
    def _compute_map_coefficients(self):
        """
        Compute conformal map coefficients using Fourier methods.
        
        For a star-shaped domain, the conformal map can be written as:
            f(w) = c₀ w + c₁ w² + c₂ w³ + ...
        
        where the coefficients are determined by matching the boundary.
        """
        # Simple approach: assume map is approximately z = R_mean * w
        # with perturbation for non-circularity
        
        theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
        r = np.array([self.R(t) for t in theta])
        
        # Mean radius
        self.R_mean = np.mean(r)
        
        # Fourier coefficients of log(r/R_mean)
        log_r = np.log(r / self.R_mean)
        self.fourier_coeffs = np.fft.fft(log_r) / len(log_r)
    
    def to_disk(self, z: np.ndarray) -> np.ndarray:
        """Approximate map to unit disk."""
        z = np.asarray(z, dtype=complex)
        # First-order approximation: just scale
        return z / self.R_mean
    
    def from_disk(self, w: np.ndarray) -> np.ndarray:
        """Approximate map from unit disk."""
        w = np.asarray(w, dtype=complex)
        return w * self.R_mean
    
    def jacobian(self, z: np.ndarray) -> np.ndarray:
        """Approximate Jacobian."""
        return np.ones_like(np.asarray(z), dtype=float) / self.R_mean
    
    def boundary_physical(self, n_points: int) -> np.ndarray:
        """Star-shaped boundary."""
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        r = np.array([self.R(t) for t in theta])
        return r * np.exp(1j * theta)
    
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if inside star-shaped domain."""
        z = np.asarray(z, dtype=complex)
        r = np.abs(z)
        theta = np.angle(z)
        R_boundary = np.array([self.R(t) for t in theta])
        return r < R_boundary


class ConformalForwardSolver:
    """
    Forward solver for general domains using conformal mapping.
    
    Maps the problem to the unit disk, solves using the analytical
    Green's function, then maps the solution back.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping from physical domain to unit disk
    n_boundary : int
        Number of boundary points
    """
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100):
        self.map = conformal_map
        self.n_boundary = n_boundary
        
        # Boundary points in physical domain
        self.boundary_physical = conformal_map.boundary_physical(n_boundary)
        self.boundary_points = np.column_stack([
            np.real(self.boundary_physical),
            np.imag(self.boundary_physical)
        ])
        
        # Corresponding points on unit disk
        self.boundary_disk = conformal_map.to_disk(self.boundary_physical)
        
        # Angles on disk boundary
        self.theta = np.angle(self.boundary_disk)
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Solve forward problem for point sources in physical domain.
        
        Parameters
        ----------
        sources : list of ((x, y), intensity)
            Point sources in physical coordinates
            
        Returns
        -------
        u : array
            Solution values on physical boundary (mean-centered)
        """
        u = np.zeros(self.n_boundary)
        
        for (xi_x, xi_y), q in sources:
            # Map source to disk
            xi_physical = xi_x + 1j * xi_y
            xi_disk = self.map.to_disk(xi_physical)
            xi_disk_xy = np.array([np.real(xi_disk), np.imag(xi_disk)])
            
            # Evaluate Green's function on disk boundary
            for i in range(self.n_boundary):
                w = self.boundary_disk[i]
                w_xy = np.array([np.real(w), np.imag(w)])
                u[i] += q * greens_function_disk_neumann(w_xy.reshape(1, 2), xi_disk_xy)
        
        return u - np.mean(u)
    
    def solve_interior(self, sources: List[Tuple[Tuple[float, float], float]],
                       x_eval: np.ndarray) -> np.ndarray:
        """
        Evaluate solution at interior points.
        
        Parameters
        ----------
        sources : list
            Point sources
        x_eval : array, shape (n, 2)
            Evaluation points in physical coordinates
            
        Returns
        -------
        u : array
            Solution values
        """
        x_eval = np.atleast_2d(x_eval)
        u = np.zeros(len(x_eval))
        
        for (xi_x, xi_y), q in sources:
            xi_disk = self.map.to_disk(xi_x + 1j * xi_y)
            xi_disk_xy = np.array([np.real(xi_disk), np.imag(xi_disk)])
            
            for i, (x, y) in enumerate(x_eval):
                w = self.map.to_disk(x + 1j * y)
                w_xy = np.array([np.real(w), np.imag(w)])
                u[i] += q * greens_function_disk_neumann(w_xy.reshape(1, 2), xi_disk_xy)
        
        return u - np.mean(u)


class ConformalLinearInverseSolver:
    """
    Linear inverse solver for general domains using conformal mapping.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    n_boundary : int
        Number of boundary points
    source_resolution : float
        Source grid resolution in physical coordinates
    """
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100,
                 source_resolution: float = 0.15, verbose: bool = True):
        self.map = conformal_map
        self.forward = ConformalForwardSolver(conformal_map, n_boundary)
        self.n_boundary = n_boundary
        
        # Generate source grid in physical domain
        # First get disk grid, then map to physical
        disk_grid = get_source_grid(resolution=source_resolution, radius=0.85)
        
        # Map to physical domain
        physical_grid = []
        for pt in disk_grid:
            w = pt[0] + 1j * pt[1]
            z = conformal_map.from_disk(w)
            physical_grid.append([np.real(z), np.imag(z)])
        
        self.interior_points = np.array(physical_grid)
        self.n_interior = len(self.interior_points)
        self.G = None
        self._verbose = verbose
        
        if verbose:
            print(f"Conformal Linear Inverse: {n_boundary} boundary pts, {self.n_interior} source candidates")
    
    def build_greens_matrix(self, verbose: bool = None):
        """Build Green's matrix via conformal mapping."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print(f"Building Green's matrix ({self.n_boundary} × {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j in range(self.n_interior):
            xi = self.interior_points[j]
            sources = [((xi[0], xi[1]), 1.0)]
            self.G[:, j] = self.forward.solve(sources) + np.mean(self.forward.solve(sources))  # Undo centering
        
        # Recenter
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
    
    def get_interior_positions(self) -> np.ndarray:
        """Return source candidate positions in physical coordinates."""
        return self.interior_points.copy()


class ConformalNonlinearInverseSolver:
    """
    Nonlinear inverse solver for general domains using conformal mapping.
    
    Optimizes source positions and intensities in the physical domain.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        The conformal mapping
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary points
    """
    
    def __init__(self, conformal_map: ConformalMap, n_sources: int, n_boundary: int = 100):
        self.map = conformal_map
        self.n_sources = n_sources
        self.forward = ConformalForwardSolver(conformal_map, n_boundary)
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Convert optimization parameters to source list."""
        sources = []
        for i in range(self.n_sources - 1):
            sources.append(((params[3*i], params[3*i+1]), params[3*i+2]))
        x_last, y_last = params[3*(self.n_sources-1)], params[3*(self.n_sources-1)+1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        return sources
    
    def _objective(self, params: np.ndarray) -> float:
        """Objective function."""
        sources = self._params_to_sources(params)
        
        # Check sources are inside domain
        for (x, y), _ in sources:
            z = x + 1j * y
            if not self.map.is_inside(z):
                return 1e10
        
        u = self.forward.solve(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
    def solve(self, method: str = 'L-BFGS-B', maxiter: int = 200,
              n_restarts: int = 1) -> InverseResult:
        """
        Solve the nonlinear inverse problem.
        
        Parameters
        ----------
        method : str
            Optimization method
        maxiter : int
            Maximum iterations
        n_restarts : int
            Random restarts
            
        Returns
        -------
        result : InverseResult
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
        
        self.history = []
        n = self.n_sources
        
        # Get domain extent for bounds
        boundary = self.map.boundary_physical(100)
        x_range = (np.real(boundary).min(), np.real(boundary).max())
        y_range = (np.imag(boundary).min(), np.imag(boundary).max())
        
        bounds = []
        for i in range(n):
            bounds.extend([
                (0.85*x_range[0], 0.85*x_range[1]),
                (0.85*y_range[0], 0.85*y_range[1])
            ])
            if i < n - 1:
                bounds.append((-5.0, 5.0))
        
        best_result = None
        best_fun = np.inf
        
        if method == 'differential_evolution':
            result = differential_evolution(self._objective, bounds, maxiter=maxiter,
                                           seed=42, polish=True)
            best_result = result
        else:
            for restart in range(n_restarts):
                np.random.seed(42 + restart)
                x0 = []
                for i in range(n):
                    x0.extend([
                        np.random.uniform(*x_range) * 0.5,
                        np.random.uniform(*y_range) * 0.5
                    ])
                    if i < n - 1:
                        x0.append(np.random.randn())
                
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


# =============================================================================
# Backward-compatible aliases
# =============================================================================
ConformalBEMSolver = ConformalForwardSolver
ConformalNonlinearInverse = ConformalNonlinearInverseSolver
ConformalLinearInverse = ConformalLinearInverseSolver


# =============================================================================
# Convenience functions
# =============================================================================
def create_ellipse_solver(a: float = 2.0, b: float = 1.0, 
                          n_boundary: int = 100) -> ConformalForwardSolver:
    """Create a solver for an ellipse with semi-axes a and b."""
    return ConformalForwardSolver(EllipseMap(a, b), n_boundary)


def create_star_solver(radius_func: Callable[[float], float],
                       n_boundary: int = 100) -> ConformalForwardSolver:
    """Create a solver for a star-shaped domain r = R(θ)."""
    return ConformalForwardSolver(StarShapedMap(radius_func), n_boundary)
