"""
Analytical Solver for Poisson Equation with Point Sources on the Unit Disk
===========================================================================

This module implements forward and inverse solvers using the EXACT analytical
Neumann Green's function for the unit disk. This is NOT a Boundary Element Method
(no numerical integration is performed) - it directly evaluates the closed-form
solution derived by Shataer (2021).

Problem:
    -Δu = Σₖ qₖ δ(x - ξₖ)   in Ω = {|x| < 1}
    ∂u/∂n = 0              on ∂Ω
    
Solution (on boundary):
    u(x)|_∂Ω = (1/π) Σₖ qₖ log|x - ξₖ|² + const   (for +Δu = f convention)
    u(x)|_∂Ω = Σₖ qₖ G(x, ξₖ) + const              (for -Δu = f convention)

where the Neumann Green's function is:
    G(x, ξ) = -1/(2π) [ln|x-ξ| + ln|x-ξ*| - ln|ξ|]
    ξ* = ξ/|ξ|² is the Kelvin reflection (image point)

The compatibility condition Σₖ qₖ = 0 must be satisfied.

Key advantages:
    - EXACT solution (no discretization error in forward problem)
    - Source positions are truly continuous (no mesh required)
    - Very fast evaluation

Limitations:
    - Only works for the unit disk (use conformal mapping for other domains)
    - Only for point sources (use BEM for distributed sources)

References:
    - Shataer, S. "Inverse Source Localization" (2021) - derivation of closed form
    - Stakgold, I. "Green's Functions and Boundary Value Problems"
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Import shared mesh for source candidate grid
try:
    from .mesh import create_disk_mesh, get_source_grid
except ImportError:
    from mesh import create_disk_mesh, get_source_grid


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
    Exact Neumann Green's function for the unit disk.
    
    This is the closed-form solution derived using the method of images.
    For a source at ξ inside the unit disk, place an image source at
    ξ* = ξ/|ξ|² (outside the disk) with the SAME sign to satisfy ∂G/∂n = const.
    
    G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
    
    On the boundary |x| = 1, this simplifies using |x - ξ*| = |x - ξ|/|ξ|.
    
    Parameters
    ----------
    x : array, shape (n, 2) or (2,)
        Evaluation points
    xi : array, shape (2,)
        Source location (must be inside unit disk)
        
    Returns
    -------
    G : array, shape (n,) or scalar
        Green's function values
        
    Notes
    -----
    This Green's function satisfies:
    - ΔG = -δ(x - ξ)  inside the disk
    - ∂G/∂n = -1/(2π) = constant on the boundary
    - ∫_∂Ω ∂G/∂n ds = -1 (integrates to source strength)
    """
    x = np.atleast_2d(x)
    xi = np.asarray(xi).flatten()
    
    # Distance to source
    dx = x[:, 0] - xi[0]
    dy = x[:, 1] - xi[1]
    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-14)  # Avoid log(0)
    
    xi_norm_sq = xi[0]**2 + xi[1]**2
    
    if xi_norm_sq < 1e-14:
        # Source at origin - no image needed (or image at infinity)
        G = -1/(2*np.pi) * np.log(r)
    else:
        # Image point via Kelvin reflection
        xi_star = xi / xi_norm_sq
        dx_star = x[:, 0] - xi_star[0]
        dy_star = x[:, 1] - xi_star[1]
        r_star = np.sqrt(dx_star**2 + dy_star**2)
        r_star = np.maximum(r_star, 1e-14)
        
        G = -1/(2*np.pi) * (np.log(r) + np.log(r_star) - np.log(np.sqrt(xi_norm_sq)))
    
    return G.flatten() if len(G) > 1 else float(G[0])


def greens_function_disk_neumann_gradient(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    Gradient of Neumann Green's function with respect to source position ξ.
    
    ∂G/∂ξ is needed for gradient-based optimization in the inverse problem.
    
    Parameters
    ----------
    x : array, shape (n, 2)
        Evaluation points
    xi : array, shape (2,)
        Source location
        
    Returns
    -------
    dG_dxi : array, shape (n, 2)
        Gradient [∂G/∂ξ₁, ∂G/∂ξ₂] at each evaluation point
    """
    x = np.atleast_2d(x)
    xi = np.asarray(xi).flatten()
    n = len(x)
    
    dx = x[:, 0] - xi[0]
    dy = x[:, 1] - xi[1]
    r_sq = dx**2 + dy**2
    r_sq = np.maximum(r_sq, 1e-28)
    
    xi_norm_sq = xi[0]**2 + xi[1]**2
    
    # Gradient of ln|x - ξ| w.r.t. ξ
    dlogr_dxi = np.column_stack([dx / r_sq, dy / r_sq])  # Note: positive because ∂/∂ξ of (x-ξ)
    
    if xi_norm_sq < 1e-14:
        dG_dxi = (1/(2*np.pi)) * dlogr_dxi
    else:
        # Image point
        xi_star = xi / xi_norm_sq
        dx_star = x[:, 0] - xi_star[0]
        dy_star = x[:, 1] - xi_star[1]
        r_star_sq = dx_star**2 + dy_star**2
        r_star_sq = np.maximum(r_star_sq, 1e-28)
        
        # ∂ξ*/∂ξ (Jacobian of Kelvin transform)
        # ξ* = ξ/|ξ|², so ∂ξ*ᵢ/∂ξⱼ = (δᵢⱼ|ξ|² - 2ξᵢξⱼ)/|ξ|⁴
        J = (np.eye(2) * xi_norm_sq - 2 * np.outer(xi, xi)) / (xi_norm_sq**2)
        
        # ∂ln|x-ξ*|/∂ξ = (∂ln|x-ξ*|/∂ξ*) @ (∂ξ*/∂ξ)
        dlogr_star_dxi_star = np.column_stack([dx_star / r_star_sq, dy_star / r_star_sq])
        dlogr_star_dxi = dlogr_star_dxi_star @ J
        
        # ∂ln|ξ|/∂ξ = ξ/|ξ|²
        dlog_xi_dxi = xi / xi_norm_sq
        
        dG_dxi = (1/(2*np.pi)) * (dlogr_dxi + dlogr_star_dxi - dlog_xi_dxi)
    
    return dG_dxi


class AnalyticalForwardSolver:
    """
    Forward solver using the exact analytical Green's function for the unit disk.
    
    This evaluates u(x) = Σₖ qₖ G(x, ξₖ) directly using the closed-form
    Green's function - no discretization or numerical integration.
    
    Parameters
    ----------
    n_boundary_points : int
        Number of points on the boundary where solution is evaluated
        
    Examples
    --------
    >>> sources = [((-0.3, 0.4), 1.0), ((0.3, -0.4), -1.0)]
    >>> solver = AnalyticalForwardSolver(n_boundary_points=100)
    >>> u = solver.solve(sources)
    """
    
    def __init__(self, n_boundary_points: int = 100):
        self.n_boundary = n_boundary_points
        self.theta = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
        self.boundary_points = np.column_stack([np.cos(self.theta), np.sin(self.theta)])
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute boundary values for given point sources.
        
        Parameters
        ----------
        sources : list of ((x, y), intensity)
            Point sources inside the unit disk
            
        Returns
        -------
        u : array, shape (n_boundary,)
            Solution values on the boundary (mean-centered)
        """
        # Check compatibility condition
        total_q = sum(q for _, q in sources)
        if abs(total_q) > 1e-10:
            print(f"Warning: Σqₖ = {total_q:.6e} ≠ 0 (compatibility condition violated)")
        
        u = np.zeros(self.n_boundary)
        for (xi_x, xi_y), q in sources:
            xi = np.array([xi_x, xi_y])
            if xi_x**2 + xi_y**2 >= 1.0:
                print(f"Warning: Source at ({xi_x:.3f}, {xi_y:.3f}) is outside or on boundary")
                continue
            u += q * greens_function_disk_neumann(self.boundary_points, xi)
        
        return u - np.mean(u)
    
    def solve_interior(self, sources: List[Tuple[Tuple[float, float], float]], 
                       x_eval: np.ndarray) -> np.ndarray:
        """
        Compute solution at interior points.
        
        Parameters
        ----------
        sources : list
            Point sources
        x_eval : array, shape (n, 2)
            Interior evaluation points
            
        Returns
        -------
        u : array, shape (n,)
            Solution values (mean-centered)
        """
        x_eval = np.atleast_2d(x_eval)
        u = np.zeros(len(x_eval))
        for (xi_x, xi_y), q in sources:
            u += q * greens_function_disk_neumann(x_eval, np.array([xi_x, xi_y]))
        return u - np.mean(u)
    
    def solve_with_gradient(self, sources: List[Tuple[Tuple[float, float], float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute boundary values and gradients with respect to source positions.
        
        Returns
        -------
        u : array, shape (n_boundary,)
            Boundary values
        grad_u : array, shape (n_sources, n_boundary, 2)
            Gradient of u w.r.t. each source position
        """
        u = np.zeros(self.n_boundary)
        grad_u = np.zeros((len(sources), self.n_boundary, 2))
        
        for k, ((xi_x, xi_y), q) in enumerate(sources):
            xi = np.array([xi_x, xi_y])
            u += q * greens_function_disk_neumann(self.boundary_points, xi)
            grad_u[k] = q * greens_function_disk_neumann_gradient(self.boundary_points, xi)
        
        return u - np.mean(u), grad_u


class AnalyticalLinearInverseSolver:
    """
    Linear inverse solver using the analytical Green's function.
    
    Places source candidates on a grid inside the domain and solves for
    intensities q ∈ ℝⁿ using the linear system Gq = u with regularization.
    
    Parameters
    ----------
    n_boundary : int
        Number of boundary measurement points
    source_resolution : float
        Grid spacing for source candidates (larger = fewer candidates)
    verbose : bool
        Print information
        
    Notes
    -----
    This is a linear formulation because source positions are fixed.
    The nonlinear formulation (AnalyticalNonlinearInverseSolver) allows
    continuous source positions but is non-convex.
    """
    
    def __init__(self, n_boundary: int = 100, source_resolution: float = 0.15, verbose: bool = True):
        self.n_boundary = n_boundary
        self.theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        self.boundary_points = np.column_stack([
            np.cos(self.theta_boundary), 
            np.sin(self.theta_boundary)
        ])
        
        # Use shared mesh for source candidates
        self.interior_points = get_source_grid(resolution=source_resolution, radius=0.9)
        self.n_interior = len(self.interior_points)
        self.G = None
        self._verbose = verbose
        
        if verbose:
            print(f"Analytical Linear Inverse: {n_boundary} boundary pts, {self.n_interior} source candidates")
        
    def build_greens_matrix(self, verbose: bool = None):
        """
        Build the Green's matrix G where G[i,j] = G(x_i, ξ_j).
        
        This matrix maps source intensities to boundary measurements: u = G @ q
        """
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print(f"Building Green's matrix ({self.n_boundary} × {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        for j in range(self.n_interior):
            self.G[:, j] = greens_function_disk_neumann(self.boundary_points, self.interior_points[j])
        
        # Center columns (remove mean from each column)
        self.G = self.G - np.mean(self.G, axis=0, keepdims=True)
        
        if verbose:
            print("Done.")
    
    def solve_l2(self, u_measured: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
        """
        Solve with Tikhonov (L2) regularization.
        
        Minimizes: ||Gq - u||² + α||q||²
        
        Parameters
        ----------
        u_measured : array
            Boundary measurements
        alpha : float
            Regularization parameter
            
        Returns
        -------
        q : array
            Recovered source intensities (mean-centered)
        """
        if self.G is None: 
            self.build_greens_matrix()
        u = u_measured - np.mean(u_measured)
        q = np.linalg.solve(self.G.T @ self.G + alpha * np.eye(self.n_interior), self.G.T @ u)
        return q - np.mean(q)
    
    def solve_l1(self, u_measured: np.ndarray, alpha: float = 1e-4, max_iter: int = 50) -> np.ndarray:
        """
        Solve with L1 (sparsity-promoting) regularization via IRLS.
        
        Minimizes: ||Gq - u||² + α||q||₁
        
        This promotes sparse solutions, which is natural for point source recovery.
        
        Parameters
        ----------
        u_measured : array
            Boundary measurements
        alpha : float
            Regularization parameter
        max_iter : int
            Maximum IRLS iterations
            
        Returns
        -------
        q : array
            Recovered source intensities (mean-centered, sparse)
        """
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
        
        Minimizes: ||Gq - u||² + α·TV(q)
        
        where TV(q) = Σ|∇q| summed over mesh edges.
        
        Parameters
        ----------
        u_measured : array
            Boundary measurements
        alpha : float
            Regularization parameter
        method : str
            'admm' or 'chambolle_pock'
        rho : float
            ADMM penalty parameter
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
        
        # Build gradient operator on triangulated source mesh
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
            try:
                from .regularization import solve_tv_chambolle_pock
            except ImportError:
                from regularization import solve_tv_chambolle_pock
            result = solve_tv_chambolle_pock(self.G, u, D, alpha=alpha, 
                                             max_iter=max_iter, verbose=verbose)
            q = result.q
        
        else:
            raise ValueError(f"Unknown TV method: {method}. Use 'admm' or 'chambolle_pock'")
        
        return q - np.mean(q)
    
    def get_interior_positions(self) -> np.ndarray:
        """Return source candidate grid positions."""
        return self.interior_points.copy()


class AnalyticalNonlinearInverseSolver:
    """
    Nonlinear inverse solver with truly continuous source positions.
    
    Optimizes both positions ξ and intensities q by minimizing the
    data misfit ||u(ξ, q) - u_measured||².
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary measurement points
        
    Notes
    -----
    This is a non-convex optimization problem with many local minima.
    Global optimization methods (differential_evolution, basinhopping)
    are recommended over local methods (L-BFGS-B).
    
    The compatibility condition Σq = 0 is enforced by parameterizing
    only n-1 intensities and computing the last as q_n = -Σq_{k<n}.
    """
    
    def __init__(self, n_sources: int, n_boundary: int = 100):
        self.n_sources = n_sources
        self.forward = AnalyticalForwardSolver(n_boundary)
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Convert optimization parameters to source list."""
        sources = []
        for i in range(self.n_sources - 1):
            sources.append(((params[3*i], params[3*i+1]), params[3*i+2]))
        # Last source: position from params, intensity from compatibility
        x_last, y_last = params[3*(self.n_sources-1)], params[3*(self.n_sources-1)+1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        return sources
    
    def _objective(self, params: np.ndarray) -> float:
        """Objective function: ||u_forward - u_measured||²"""
        sources = self._params_to_sources(params)
        
        # Check all sources are inside the disk
        for (x, y), _ in sources:
            if x**2 + y**2 >= 0.9**2: 
                return 1e10  # Penalty for sources outside domain
        
        u = self.forward.solve(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
    def _get_initial_guess(self, init_from: str, seed: int) -> List[float]:
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
            # Symmetric initial guess on circle
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
            Optimization method:
            - 'L-BFGS-B': Local quasi-Newton (fast, may get stuck)
            - 'differential_evolution': Global stochastic (slower, more robust)
            - 'basinhopping': Global with local polish
            - 'SLSQP': Local SQP method
        maxiter : int
            Maximum iterations
        n_restarts : int
            Number of random restarts for local optimizers
        init_from : str
            Initial guess type: 'circle' or 'random'
            
        Returns
        -------
        result : InverseResult
            Recovered sources, residual, and optimization info
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
            
        self.history = []
        n = self.n_sources
        
        # Bounds: positions in disk, intensities unbounded but reasonable
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
            # Local optimizer with restarts
            for restart in range(n_restarts):
                x0 = self._get_initial_guess(init_from, restart)
                result = minimize(self._objective, x0, method=method, bounds=bounds, 
                                options={'maxiter': maxiter})
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
        
        # Convert result to Source objects
        sources = [Source(x, y, q) for (x, y), q in self._params_to_sources(best_result.x)]
        
        return InverseResult(
            sources=sources,
            residual=np.sqrt(best_result.fun),
            success=best_result.success if hasattr(best_result, 'success') else True,
            message=str(best_result.message) if hasattr(best_result, 'message') else '',
            iterations=best_result.nit if hasattr(best_result, 'nit') else len(self.history),
            history=self.history
        )


def generate_synthetic_data(sources: List[Tuple[Tuple[float, float], float]], 
                           n_boundary: int = 100, 
                           noise_level: float = 0.0, 
                           seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic boundary measurements for testing.
    
    Parameters
    ----------
    sources : list
        True source configuration
    n_boundary : int
        Number of boundary points
    noise_level : float
        Standard deviation of additive Gaussian noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    theta : array
        Boundary angles
    u : array
        Noisy boundary measurements
    """
    if seed is not None: 
        np.random.seed(seed)
    forward = AnalyticalForwardSolver(n_boundary)
    u = forward.solve(sources)
    if noise_level > 0: 
        u += np.random.normal(0, noise_level, len(u))
    return forward.theta, u


# =============================================================================
# Backward-compatible aliases (deprecated - use new names)
# =============================================================================
BEMForwardSolver = AnalyticalForwardSolver
BEMLinearInverseSolver = AnalyticalLinearInverseSolver
BEMNonlinearInverseSolver = AnalyticalNonlinearInverseSolver
