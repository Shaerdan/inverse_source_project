"""
Boundary Element Method Solver for Poisson Equation with Point Sources
=======================================================================

This module implements BEM-based forward and inverse solvers using the
analytical Neumann Green's function for the unit disk.

Key advantages:
    - Source positions are truly continuous (no mesh required for forward)
    - Analytical Green's function = fast and accurate
    - Source candidate grid uses same uniform mesh type as FEM (for consistency)

References:
    - Stakgold, I. "Green's Functions and Boundary Value Problems"
    - Sauter & Schwab, "Boundary Element Methods"
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import shared mesh for source candidate grid
from .mesh import create_disk_mesh, get_source_grid


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


def greens_function_disk_neumann(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Neumann Green's function for the unit disk.
    
    G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
    
    where ξ* = ξ/|ξ|² is the image point.
    """
    x = np.atleast_2d(x)
    xi = np.asarray(xi).flatten()
    
    dx = x[:, 0] - xi[0]
    dy = x[:, 1] - xi[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-14)
    
    xi_norm_sq = xi[0]**2 + xi[1]**2
    
    if xi_norm_sq < 1e-14:
        G = -1/(2*np.pi) * np.log(r)
    else:
        xi_star = xi / xi_norm_sq
        dx_star = x[:, 0] - xi_star[0]
        dy_star = x[:, 1] - xi_star[1]
        r_star = np.sqrt(dx_star**2 + dy_star**2)
        r_star = np.maximum(r_star, 1e-14)
        
        G = -1/(2*np.pi) * (np.log(r) + np.log(r_star) - np.log(np.sqrt(xi_norm_sq)))
    
    return G.flatten() if len(G) > 1 else float(G[0])


class BEMForwardSolver:
    """Forward solver using analytical Green's function for the unit disk."""
    
    def __init__(self, n_boundary_points: int = 100):
        self.n_boundary = n_boundary_points
        self.theta = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
        self.boundary_points = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """Compute boundary values for given sources."""
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6e} ≠ 0")
        
        u = np.zeros(self.n_boundary)
        for (xi_x, xi_y), q in sources:
            xi = np.array([xi_x, xi_y])
            if xi_x**2 + xi_y**2 >= 1.0:
                print(f"Warning: Source outside disk")
                continue
            u += q * greens_function_disk_neumann(self.boundary_points, xi)
        
        return u - np.mean(u)
    
    def solve_interior(self, sources, x_eval: np.ndarray) -> np.ndarray:
        """Compute solution at interior points."""
        x_eval = np.atleast_2d(x_eval)
        u = np.zeros(len(x_eval))
        for (xi_x, xi_y), q in sources:
            u += q * greens_function_disk_neumann(x_eval, np.array([xi_x, xi_y]))
        return u - np.mean(u)


class BEMLinearInverseSolver:
    """
    Linear inverse solver with L1, L2, and TV regularization.
    
    Uses uniform triangular mesh (same as FEM) for source candidate grid.
    
    Parameters
    ----------
    n_boundary : int
        Number of boundary measurement points
    source_resolution : float
        Mesh resolution for source candidate grid (larger = coarser = fewer candidates)
    verbose : bool
        Print info. Default True
    """
    
    def __init__(self, n_boundary: int = 100, source_resolution: float = 0.15, verbose: bool = True):
        self.n_boundary = n_boundary
        self.theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        self.boundary_points = np.column_stack([
            np.cos(self.theta_boundary), 
            np.sin(self.theta_boundary)
        ])
        
        # Use shared mesh for source candidates (same type as FEM)
        self.interior_points = get_source_grid(resolution=source_resolution, radius=0.9)
        self.n_interior = len(self.interior_points)
        self.G = None
        self._verbose = verbose
        
        if verbose:
            print(f"BEM Linear: {n_boundary} boundary, {self.n_interior} source candidates")
        
    def build_greens_matrix(self, verbose: bool = None):
        """Build the Green's matrix analytically."""
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print(f"Building Green's matrix ({self.n_boundary} x {self.n_interior})...")
        self.G = np.zeros((self.n_boundary, self.n_interior))
        for j in range(self.n_interior):
            self.G[:, j] = greens_function_disk_neumann(self.boundary_points, self.interior_points[j])
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
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, max_iter: int = 50) -> np.ndarray:
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
        u = u_measured - np.mean(u_measured)
        
        # Build gradient on triangulated source mesh
        tri = Delaunay(self.interior_points)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                edges.add(tuple(sorted([s[i], s[(i+1)%3]])))
        
        D = np.zeros((len(edges), self.n_interior))
        for k, (i, j) in enumerate(edges):
            D[k, i], D[k, j] = 1, -1
        
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
        """Return source candidate grid positions."""
        return self.interior_points.copy()


class BEMNonlinearInverseSolver:
    """Nonlinear inverse solver with truly continuous source positions."""
    
    def __init__(self, n_sources: int, n_boundary: int = 100):
        self.n_sources = n_sources
        self.forward = BEMForwardSolver(n_boundary)
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        self.u_measured = u_measured - np.mean(u_measured)
    
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
            if x**2 + y**2 >= 0.9**2: 
                return 1e10
        u = self.forward.solve(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
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
            for i in range(n):
                angle = 2 * np.pi * i / n
                x0.extend([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
                if i < n - 1: 
                    x0.append(1.0 if i % 2 == 0 else -1.0)
        
        return x0
    
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
            Number of random restarts for local optimizers
        init_from : str
            'circle' or 'random'
        """
        self.history = []
        n = self.n_sources
        
        bounds = []
        for i in range(n):
            bounds.extend([(-0.85, 0.85), (-0.85, 0.85)])
            if i < n - 1: 
                bounds.append((-5.0, 5.0))
        
        best_result = None
        best_fun = np.inf
        
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
            for restart in range(n_restarts):
                x0 = self._get_initial_guess(init_from, restart)
                result = minimize(self._objective, x0, method=method, bounds=bounds, 
                                options={'maxiter': maxiter})
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
        
        sources = [Source(x, y, q) for (x, y), q in self._params_to_sources(best_result.x)]
        return InverseResult(
            sources, np.sqrt(best_result.fun), 
            best_result.success if hasattr(best_result, 'success') else True,
            str(best_result.message) if hasattr(best_result, 'message') else '', 
            best_result.nit if hasattr(best_result, 'nit') else len(self.history), 
            self.history
        )


def generate_synthetic_data(sources, n_boundary=100, noise_level=0.0, seed=None):
    """Generate synthetic boundary measurements."""
    if seed: 
        np.random.seed(seed)
    forward = BEMForwardSolver(n_boundary)
    u = forward.solve(sources)
    if noise_level > 0: 
        u += np.random.normal(0, noise_level, len(u))
    return forward.theta, u
