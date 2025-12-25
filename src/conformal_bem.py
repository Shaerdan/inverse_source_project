"""
Conformal BEM Solver for General Simply Connected Domains
==========================================================

Uses conformal mapping to transform problems on general domains to the unit disk.

Key insight: G_Ω(z₁, z₂) = G_D(f(z₁), f(z₂)) where f: Ω → D is conformal.

Supported domains:
    - Unit disk (identity map)
    - Ellipse (Joukowsky transform)
    - Star-shaped domains (numerical map)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize, differential_evolution
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .bem_solver import greens_function_disk_neumann, Source, InverseResult


class ConformalMap(ABC):
    """Abstract base class for conformal maps from domain Ω to unit disk D."""
    
    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        """Map from physical domain Ω to unit disk D: w = f(z)"""
        pass
    
    @abstractmethod
    def inverse(self, w: np.ndarray) -> np.ndarray:
        """Map from unit disk D to physical domain Ω: z = f⁻¹(w)"""
        pass
    
    @abstractmethod
    def boundary_physical(self, n_points: int = 100) -> np.ndarray:
        """Return points on the physical domain boundary (as complex numbers)."""
        pass
    
    @abstractmethod
    def is_inside(self, z: np.ndarray) -> np.ndarray:
        """Check if point(s) z are inside the physical domain."""
        pass


class DiskMap(ConformalMap):
    """Identity map for the unit disk (trivial case)."""
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def forward(self, z):
        return np.asarray(z) / self.radius
    
    def inverse(self, w):
        return np.asarray(w) * self.radius
    
    def boundary_physical(self, n_points=100):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.radius * np.exp(1j * theta)
    
    def is_inside(self, z):
        return np.abs(z) < self.radius


class EllipseMap(ConformalMap):
    """
    Conformal map for an ellipse with semi-axes a, b.
    
    Uses the Joukowsky-type transformation:
        z = (a+b)/2 * w + (a-b)/2 * 1/w
    """
    
    def __init__(self, a: float = 2.0, b: float = 1.0):
        if a < b: a, b = b, a
        self.a, self.b = a, b
        self.c1 = (a + b) / 2
        self.c2 = (a - b) / 2
    
    def forward(self, z):
        z = np.asarray(z, dtype=complex)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)
        
        disc = z**2 - 4 * self.c1 * self.c2
        sqrt_disc = np.sqrt(disc)
        w1 = (z + sqrt_disc) / (2 * self.c1)
        w2 = (z - sqrt_disc) / (2 * self.c1)
        w = np.where(np.abs(w1) < np.abs(w2), w1, w2)
        
        return w.item() if scalar else w
    
    def inverse(self, w):
        w = np.asarray(w, dtype=complex)
        w_safe = np.where(np.abs(w) < 1e-14, 1e-14, w)
        return self.c1 * w + self.c2 / w_safe
    
    def boundary_physical(self, n_points=100):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.a * np.cos(theta) + 1j * self.b * np.sin(theta)
    
    def is_inside(self, z):
        z = np.asarray(z, dtype=complex)
        return (np.real(z)/self.a)**2 + (np.imag(z)/self.b)**2 < 1


class StarShapedMap(ConformalMap):
    """
    Conformal map for a star-shaped domain defined by r(θ).
    
    Uses a simple radial scaling approximation (accurate for nearly circular domains).
    """
    
    def __init__(self, r_func: Callable[[np.ndarray], np.ndarray]):
        self.r_func = r_func
        theta = np.linspace(0, 2*np.pi, 256, endpoint=False)
        self.r_interp = interp1d(theta, r_func(theta), kind='cubic', fill_value='extrapolate')
    
    def forward(self, z):
        z = np.asarray(z, dtype=complex)
        theta = np.angle(z) % (2*np.pi)
        r_at_theta = self.r_interp(theta)
        return z / r_at_theta
    
    def inverse(self, w):
        w = np.asarray(w, dtype=complex)
        theta = np.angle(w) % (2*np.pi)
        r_at_theta = self.r_interp(theta)
        return r_at_theta * w
    
    def boundary_physical(self, n_points=100):
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return self.r_func(theta) * np.exp(1j * theta)
    
    def is_inside(self, z):
        z = np.asarray(z, dtype=complex)
        r = np.abs(z)
        theta = np.angle(z) % (2*np.pi)
        return r < self.r_interp(theta)


class ConformalBEMSolver:
    """
    BEM solver for general simply connected domains using conformal mapping.
    
    The Green's function transforms as: G_Ω(z₁, z₂) = G_D(f(z₁), f(z₂))
    """
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100):
        self.map = conformal_map
        self.n_boundary = n_boundary
        
        self.boundary_physical = conformal_map.boundary_physical(n_boundary)
        self.boundary_disk = conformal_map.forward(self.boundary_physical)
        self.boundary_param = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
    
    def greens_function(self, x: np.ndarray, xi: complex) -> np.ndarray:
        """Green's function for the physical domain: G_Ω(x, ξ) = G_D(f(x), f(ξ))"""
        w_x = self.map.forward(x)
        w_xi = self.map.forward(xi)
        
        if np.isscalar(w_x) or w_x.ndim == 0:
            x_disk = np.array([[np.real(w_x), np.imag(w_x)]])
        else:
            x_disk = np.column_stack([np.real(w_x), np.imag(w_x)])
        
        xi_disk = np.array([np.real(w_xi), np.imag(w_xi)])
        return greens_function_disk_neumann(x_disk, xi_disk)
    
    def solve_forward(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Solve forward problem: compute boundary values."""
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6f} ≠ 0")
        
        u = np.zeros(self.n_boundary)
        for (xi_x, xi_y), q in sources:
            xi = xi_x + 1j * xi_y
            if not self.map.is_inside(xi):
                print(f"Warning: Source at ({xi_x}, {xi_y}) outside domain")
                continue
            u += q * self.greens_function(self.boundary_physical, xi)
        
        return u - np.mean(u)
    
    def build_greens_matrix(self, interior_points: np.ndarray) -> np.ndarray:
        """Build Green's matrix for linear inverse problem."""
        n_int = len(interior_points)
        G = np.zeros((self.n_boundary, n_int))
        
        for j, pt in enumerate(interior_points):
            if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) == 2:
                xi = pt[0] + 1j * pt[1]
            else:
                xi = complex(pt)
            G[:, j] = self.greens_function(self.boundary_physical, xi)
        
        return G - np.mean(G, axis=0, keepdims=True)


class ConformalNonlinearInverse:
    """Nonlinear inverse solver for general domains."""
    
    def __init__(self, conformal_map: ConformalMap, n_sources: int, n_boundary: int = 100):
        self.solver = ConformalBEMSolver(conformal_map, n_boundary)
        self.map = conformal_map
        self.n_sources = n_sources
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u: np.ndarray):
        self.u_measured = u - np.mean(u)
    
    def _params_to_sources(self, params):
        sources = []
        for i in range(self.n_sources - 1):
            sources.append(((params[3*i], params[3*i+1]), params[3*i+2]))
        x_last, y_last = params[3*(self.n_sources-1)], params[3*(self.n_sources-1)+1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        return sources
    
    def _objective(self, params):
        sources = self._params_to_sources(params)
        for (x, y), _ in sources:
            if not self.map.is_inside(x + 1j * y):
                return 1e10
        u = self.solver.solve_forward(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
    def solve(self, method: str = 'L-BFGS-B', maxiter: int = 200) -> InverseResult:
        self.history = []
        
        # Get domain bounds
        boundary = self.solver.boundary_physical
        x_min, x_max = np.real(boundary).min(), np.real(boundary).max()
        y_min, y_max = np.imag(boundary).min(), np.imag(boundary).max()
        margin = 0.15
        x_min += margin * (x_max - x_min)
        x_max -= margin * (x_max - x_min)
        y_min += margin * (y_max - y_min)
        y_max -= margin * (y_max - y_min)
        
        bounds = []
        for i in range(self.n_sources):
            bounds.extend([(x_min, x_max), (y_min, y_max)])
            if i < self.n_sources - 1:
                bounds.append((-5.0, 5.0))
        
        # Initial guess
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        r = min(x_max - x_min, y_max - y_min) * 0.25
        x0 = []
        for i in range(self.n_sources):
            angle = 2 * np.pi * i / self.n_sources
            x0.extend([cx + r*np.cos(angle), cy + r*np.sin(angle)])
            if i < self.n_sources - 1:
                x0.append(1.0 if i % 2 == 0 else -1.0)
        
        if method == 'differential_evolution':
            result = differential_evolution(self._objective, bounds, maxiter=maxiter, seed=42)
        else:
            result = minimize(self._objective, x0, method=method, bounds=bounds, options={'maxiter': maxiter})
        
        sources = [Source(x, y, q) for (x, y), q in self._params_to_sources(result.x)]
        return InverseResult(sources, np.sqrt(result.fun), result.success,
                            str(result.message) if hasattr(result, 'message') else '',
                            result.nit if hasattr(result, 'nit') else len(self.history), self.history)


class ConformalLinearInverse:
    """Linear inverse solver for general domains."""
    
    def __init__(self, conformal_map: ConformalMap, n_boundary: int = 100, n_interior: int = 200):
        self.solver = ConformalBEMSolver(conformal_map, n_boundary)
        self.map = conformal_map
        self.interior_points = self._generate_interior_grid(n_interior)
        self.n_interior = len(self.interior_points)
        self.G = None
    
    def _generate_interior_grid(self, n_target: int) -> np.ndarray:
        """Generate interior points using rejection sampling."""
        boundary = self.solver.boundary_physical
        x_min, x_max = np.real(boundary).min(), np.real(boundary).max()
        y_min, y_max = np.imag(boundary).min(), np.imag(boundary).max()
        
        points = []
        for _ in range(n_target * 100):
            if len(points) >= n_target:
                break
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = x + 1j * y
            if self.map.is_inside(z):
                w = self.map.forward(z)
                if np.abs(w) < 0.9:
                    points.append(z)
        
        return np.array(points)
    
    def build_greens_matrix(self):
        print("Building Green's matrix...")
        self.G = self.solver.build_greens_matrix(self.interior_points)
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        if self.G is None:
            self.build_greens_matrix()
        
        u = u_measured - np.mean(u_measured)
        q, eps = np.zeros(self.n_interior), 1e-4
        GtG, Gtu = self.G.T @ self.G, self.G.T @ u
        
        for _ in range(max_iter):
            W = np.diag(1.0 / (np.abs(q) + eps))
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            if np.linalg.norm(q_new - q) < 1e-6:
                break
            q = q_new
        
        return q - np.mean(q)
