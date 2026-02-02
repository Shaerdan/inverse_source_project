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

# Import optimization utilities for multistart and interior point initialization
try:
    from .optimization_utils import (
        push_to_interior, generate_spread_init, generate_random_init,
        solve_disk_polar, boundary_potential_disk_cartesian
    )
    HAS_OPT_UTILS = True
except ImportError:
    try:
        from optimization_utils import (
            push_to_interior, generate_spread_init, generate_random_init,
            solve_disk_polar, boundary_potential_disk_cartesian
        )
        HAS_OPT_UTILS = True
    except ImportError:
        HAS_OPT_UTILS = False
        push_to_interior = None
        generate_spread_init = None
        generate_random_init = None
        solve_disk_polar = None
        boundary_potential_disk_cartesian = None


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
    
    For the PDE: -Δu = δ(x - ξ), the fundamental solution is Φ = -1/(2π) log|x-ξ|.
    This is POSITIVE near the source (since log(small r) < 0).
    
    Using the method of images for Neumann BC, the Green's function is:
    
    G(x, ξ) = -1/(2π) [ln|x - ξ| + ln|x - ξ*| - ln|ξ|]
    
    where ξ* = ξ/|ξ|² is the image point (Kelvin transform).
    
    Parameters
    ----------
    x : array, shape (n, 2) or (2,)
        Evaluation points
    xi : array, shape (2,)
        Source location (must be inside unit disk)
        
    Returns
    -------
    G : array, shape (n,) or scalar
        Green's function values (POSITIVE near source)
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
        # Source at origin - no image needed
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
        dG_dxi = (-1/(2*np.pi)) * dlogr_dxi
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
        
        dG_dxi = (-1/(2*np.pi)) * (dlogr_dxi + dlogr_star_dxi - dlog_xi_dxi)
    
    return dG_dxi


class AnalyticalForwardSolver:
    """
    Forward solver using the exact analytical Green's function for the unit disk.
    
    This evaluates u(x) = Σₖ qₖ G(x, ξₖ) directly using the closed-form
    Green's function - no discretization or numerical integration.
    
    Parameters
    ----------
    n_boundary_points : int
        Number of sensor/measurement points on the boundary.
        These are the fixed physical locations where we measure.
    sensor_locations : array, optional
        Custom sensor locations. If None, uses evenly spaced points.
        
    Examples
    --------
    >>> sources = [((-0.3, 0.4), 1.0), ((0.3, -0.4), -1.0)]
    >>> solver = AnalyticalForwardSolver(n_boundary_points=100)
    >>> u = solver.solve(sources)
    """
    
    def __init__(self, n_boundary_points: int = 100, sensor_locations: np.ndarray = None):
        if sensor_locations is not None:
            self.sensor_locations = np.asarray(sensor_locations)
            self.n_sensors = len(self.sensor_locations)
        else:
            self.n_sensors = n_boundary_points
            theta = np.linspace(0, 2*np.pi, n_boundary_points, endpoint=False)
            self.sensor_locations = np.column_stack([np.cos(theta), np.sin(theta)])
        
        # For backward compatibility
        self.n_boundary = self.n_sensors
        self.boundary_points = self.sensor_locations
        self.theta = np.arctan2(self.sensor_locations[:, 1], self.sensor_locations[:, 0])
    
    def solve(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute solution at sensor locations for given point sources.
        
        Parameters
        ----------
        sources : list of ((x, y), intensity)
            Point sources inside the unit disk
            
        Returns
        -------
        u : array, shape (n_sensors,)
            Solution values at sensor locations (mean-centered)
        """
        u = np.zeros(self.n_sensors)
        for (xi_x, xi_y), q in sources:
            xi = np.array([xi_x, xi_y])
            u += q * greens_function_disk_neumann(self.sensor_locations, xi)
        
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
        Number of sensor/measurement points on boundary
    sensor_locations : array, optional
        Custom sensor locations (n_sensors, 2). If None, uses evenly spaced.
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
    
    def __init__(self, n_boundary: int = 100, sensor_locations: np.ndarray = None,
                 source_resolution: float = 0.15, 
                 verbose: bool = True):
        
        # Set up sensor/measurement locations
        if sensor_locations is not None:
            self.sensor_locations = np.asarray(sensor_locations)
            self.n_sensors = len(self.sensor_locations)
        else:
            self.n_sensors = n_boundary
            theta = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
            self.sensor_locations = np.column_stack([np.cos(theta), np.sin(theta)])
        
        # For backward compatibility
        self.n_boundary = self.n_sensors
        self.boundary_points = self.sensor_locations
        self.theta_boundary = np.arctan2(self.sensor_locations[:, 1], self.sensor_locations[:, 0])
        
        # Use shared mesh for source candidates
        self.interior_points = get_source_grid(
            resolution=source_resolution, radius=0.9
        )
        self.n_interior = len(self.interior_points)
        self.G = None
        self._verbose = verbose
        
        if verbose:
            print(f"Analytical Linear Inverse: {self.n_sensors} sensors, {self.n_interior} source candidates")
        
    def build_greens_matrix(self, verbose: bool = None):
        """
        Build the Green's matrix G where G[i,j] = G(x_i, ξ_j).
        
        This matrix maps source intensities to sensor measurements: u = G @ q
        """
        if verbose is None:
            verbose = self._verbose
        if verbose:
            print(f"Building Green's matrix ({self.n_sensors} × {self.n_interior})...")
        
        self.G = np.zeros((self.n_sensors, self.n_interior))
        for j in range(self.n_interior):
            self.G[:, j] = greens_function_disk_neumann(self.sensor_locations, self.interior_points[j])
        
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
                 method: str = 'cvxpy', rho: float = 1.0, max_iter: int = 500,
                 tol: float = 1e-6, verbose: bool = False) -> np.ndarray:
        """
        Solve with Total Variation regularization.
        
        Minimizes: ||Gq - u||² + α·TV(q) subject to Σq = 0
        
        where TV(q) = Σ|∇q| summed over mesh edges.
        
        Parameters
        ----------
        u_measured : array
            Boundary measurements
        alpha : float
            Regularization parameter (typically 10-100x larger than L1/L2)
        method : str
            'cvxpy' (accurate, default), 'admm' (faster), or 'chambolle_pock'
        rho : float
            ADMM penalty parameter (only for method='admm')
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance for ADMM
        verbose : bool
            Print convergence info
            
        Returns
        -------
        q : array
            Recovered source intensities
            
        Notes
        -----
        TV regularization promotes piecewise-constant solutions. For point
        source recovery, α typically needs to be 10-100x larger than for L1/L2.
        Use parameter_selection.parameter_sweep() to find optimal α.
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
        
        if method.lower() == 'cvxpy':
            # Use cvxpy for accurate solution with proper constraint handling
            try:
                import cvxpy as cp
            except ImportError:
                if verbose:
                    print("cvxpy not available, falling back to ADMM")
                method = 'admm'
            else:
                n = self.n_interior
                q_var = cp.Variable(n)
                constraints = [cp.sum(q_var) == 0]  # Compatibility constraint
                objective = cp.Minimize(
                    0.5 * cp.sum_squares(self.G @ q_var - u) + alpha * cp.norm1(D @ q_var)
                )
                prob = cp.Problem(objective, constraints)
                # Use ECOS solver for consistent results across systems
                # OSQP can give different results on different machines
                try:
                    prob.solve(solver=cp.ECOS, verbose=verbose)
                except Exception:
                    prob.solve(verbose=verbose)  # Fallback to default
                
                if q_var.value is not None:
                    return q_var.value
                else:
                    if verbose:
                        print("cvxpy failed, falling back to ADMM")
                    method = 'admm'
        
        if method.lower() in ('admm', 'tv_admm'):
            # ADMM implementation with constraint projection
            n = self.n_interior
            m = len(edges)
            
            q = np.zeros(n)
            z = np.zeros(m)
            w = np.zeros(m)
            
            # Precompute matrix inverse
            A = self.G.T @ self.G + rho * D.T @ D
            A_inv = np.linalg.inv(A)
            Gtu = self.G.T @ u
            
            for it in range(max_iter):
                q_old = q.copy()
                
                # q-update
                q = A_inv @ (Gtu + rho * D.T @ (z - w))
                
                # Project onto sum(q) = 0 constraint
                q = q - np.mean(q)
                
                # z-update (soft thresholding)
                Dq = D @ q
                z_old = z.copy()
                z = np.sign(Dq + w) * np.maximum(np.abs(Dq + w) - alpha/rho, 0)
                
                # w-update (dual)
                w = w + Dq - z
                
                # Check convergence
                primal_res = np.linalg.norm(Dq - z)
                dual_res = rho * np.linalg.norm(D.T @ (z - z_old))
                
                if verbose and it % 50 == 0:
                    energy = 0.5 * np.linalg.norm(self.G @ q - u)**2 + alpha * np.sum(np.abs(D @ q))
                    print(f"  ADMM iter {it}: energy={energy:.6e}, primal={primal_res:.2e}, dual={dual_res:.2e}")
                
                if primal_res < tol and dual_res < tol:
                    if verbose:
                        print(f"  ADMM converged at iter {it}")
                    break
        
        elif method.lower() in ('chambolle_pock', 'cp', 'tv_cp'):
            try:
                from .regularization import solve_tv_chambolle_pock
            except ImportError:
                from regularization import solve_tv_chambolle_pock
            result = solve_tv_chambolle_pock(self.G, u, D, alpha=alpha, 
                                             max_iter=max_iter, verbose=verbose)
            q = result.q
        
        else:
            raise ValueError(f"Unknown TV method: {method}. Use 'cvxpy', 'admm', or 'chambolle_pock'")
        
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
        Number of sensor/measurement points
    sensor_locations : array, optional
        Custom sensor locations. If None, uses evenly spaced on unit circle.
        
    Notes
    -----
    This is a non-convex optimization problem with many local minima.
    Global optimization methods (differential_evolution, basinhopping)
    are recommended over local methods (L-BFGS-B).
    
    The compatibility condition Σq = 0 is enforced by parameterizing
    only n-1 intensities and computing the last as q_n = -Σq_{k<n}.
    """
    
    def __init__(self, n_sources: int, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None):
        self.n_sources = n_sources
        self.forward = AnalyticalForwardSolver(n_boundary, sensor_locations=sensor_locations)
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Convert optimization parameters to source list.
        
        Parameters layout: [x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, q0, q1, ..., q_{n-1}]
        (all positions first, then all intensities)
        
        Zero-sum constraint enforced by centering intensities.
        """
        n = self.n_sources
        sources = []
        
        # Extract positions
        positions = [(params[2*i], params[2*i + 1]) for i in range(n)]
        
        # Extract intensities and center them (enforces zero-sum)
        intensities = np.array([params[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)  # Centering enforces Σq = 0
        
        for i in range(n):
            sources.append((positions[i], intensities[i]))
        
        return sources
    
    def _objective_misfit(self, params: np.ndarray) -> float:
        """
        Pure misfit objective: ||u_forward - u_measured||²
        
        Used with NonlinearConstraint (DE, trust-constr) where constraint
        is handled separately.
        """
        sources = self._params_to_sources(params)
        u = self.forward.solve(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
    def _objective_with_barrier(self, params: np.ndarray, mu: float = 1e-6) -> float:
        """
        Objective with logarithmic barrier for L-BFGS-B interior point method.
        
        f(x) = ||u_forward - u_measured||² - μ * Σ log(1 - x_i² - y_i²)
        
        The log barrier enforces x² + y² < 1 (unit disk constraint).
        Gradient naturally repels optimizer from boundary.
        """
        sources = self._params_to_sources(params)
        
        # Logarithmic barrier for disk constraint: x² + y² < 1
        barrier = 0.0
        for (x, y), _ in sources:
            r_sq = x**2 + y**2
            if r_sq >= 1.0:
                return 1e12  # Outside disk
            barrier -= mu * np.log(1.0 - r_sq)
        
        u = self.forward.solve(sources)
        misfit = np.sum((u - self.u_measured)**2)
        self.history.append(misfit)
        return misfit + barrier
    
    def _disk_constraint(self, params: np.ndarray) -> np.ndarray:
        """
        Disk constraint function for NonlinearConstraint.
        
        Returns array of (1 - x_i² - y_i²) for each source.
        Constraint satisfied when all values > 0.
        """
        n = self.n_sources
        values = np.zeros(n)
        for i in range(n):
            x, y = params[2*i], params[2*i + 1]
            values[i] = 1.0 - x**2 - y**2
        return values
    
    def _get_initial_guess(self, init_from: str, seed: int) -> List[float]:
        """Generate initial guess for optimization.
        
        KEY FIX: Push initial points to interior of bounds.
        MATLAB's fmincon does this automatically - without it, linspace-style
        initializations put values at bounds, causing gradient blow-up.
        """
        n = self.n_sources
        x0 = []
        
        if init_from == 'spread':
            # Evenly spread sources around circle
            # Vary radius across restarts to cover search space (principled, not tuned)
            # seed 0->r=0.4, seed 1->r=0.55, seed 2->r=0.7, etc.
            r = 0.4 + 0.15 * (seed % 3)  # Cycles through 0.4, 0.55, 0.7
            for i in range(n):
                angle = 2 * np.pi * i / n + seed * 0.1  # Small offset per restart
                x0.extend([r * np.cos(angle), r * np.sin(angle)])
            # Alternating intensities
            for i in range(n):
                x0.append(1.0 if i % 2 == 0 else -1.0)
        elif init_from == 'random' or seed > 0:
            np.random.seed(42 + seed)
            # Positions first
            for i in range(n):
                r = 0.5 + 0.35 * np.random.rand()  # r in [0.5, 0.85]
                angle = 2 * np.pi * np.random.rand()
                x0.extend([r * np.cos(angle), r * np.sin(angle)])
            # Then intensities
            for i in range(n):
                x0.append(np.random.randn())
        else:  # 'circle' (default)
            # Symmetric initial guess on circle
            for i in range(n):
                angle = 2 * np.pi * i / n
                x0.extend([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
            # Then intensities
            for i in range(n):
                x0.append(1.0 if i % 2 == 0 else -1.0)
        
        x0 = np.array(x0)
        
        # KEY FIX: Push to interior of bounds to avoid gradient blow-up
        if HAS_OPT_UTILS and push_to_interior is not None:
            box_bounds = [(-0.95, 0.95)] * (2*n) + [(-5.0, 5.0)] * n
            x0 = push_to_interior(x0, box_bounds, margin=0.1)
        
        return list(x0)
    
    def solve(self, method: str = 'SLSQP', maxiter: int = 10000,
              n_restarts: int = 5, init_from: str = 'spread',
              mu: float = 1e-6) -> InverseResult:
        """
        Solve the nonlinear inverse problem.
        
        Parameters
        ----------
        method : str
            Optimization method:
            - 'SLSQP': Sequential Least Squares Programming (RECOMMENDED)
            - 'L-BFGS-B': Local quasi-Newton with log barrier
            - 'differential_evolution': Global with NonlinearConstraint
            - 'trust-constr': Local with NonlinearConstraint
            - 'basinhopping': Global with local polish
        maxiter : int
            Maximum iterations (default 10000 for SLSQP)
        n_restarts : int
            Number of random restarts for local optimizers (default 5)
        init_from : str
            Initial guess type: 'circle', 'spread', or 'random'
        mu : float
            Barrier parameter for L-BFGS-B interior point method
            
        Returns
        -------
        result : InverseResult
            Recovered sources, residual, and optimization info
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
            
        self.history = []
        n = self.n_sources
        
        best_result = None
        best_fun = np.inf
        
        # Box bounds: contain the disk [-0.95, 0.95] × [-0.95, 0.95], intensity [-5, 5]
        box_bounds = [(-0.95, 0.95)] * (2*n) + [(-5.0, 5.0)] * n
        
        # NonlinearConstraint: 1 - x² - y² > 0 for each source (inside disk)
        from scipy.optimize import NonlinearConstraint
        disk_constraint = NonlinearConstraint(
            self._disk_constraint, 
            0.0,      # lower bound (must be > 0, i.e., inside disk)
            np.inf    # upper bound (no upper limit)
        )
        
        # Equality constraint for SLSQP: sum of intensities = 0
        def intensity_sum(params):
            return sum(params[2*n + i] for i in range(n))
        
        if method == 'SLSQP':
            # SLSQP with equality constraint AND disk constraint
            # This matches MATLAB fmincon behavior most closely
            
            # Disk constraint: 1 - (x² + y²) >= 0 for each source
            def disk_ineq_constraint(params):
                constraints_vals = []
                for i in range(n):
                    x_i = params[2*i]
                    y_i = params[2*i + 1]
                    # 1 - r² must be >= 0 (source inside disk)
                    constraints_vals.append(1.0 - x_i**2 - y_i**2)
                return np.array(constraints_vals)
            
            constraints = [
                {'type': 'eq', 'fun': intensity_sum},  # Sum of intensities = 0
                {'type': 'ineq', 'fun': disk_ineq_constraint},  # Sources inside disk
            ]
            
            # Use diverse initialization strategies
            init_strategies = ['spread', 'circle'] + ['random'] * max(0, n_restarts - 2)
            
            for restart, init_type in enumerate(init_strategies[:n_restarts]):
                x0 = self._get_initial_guess(init_type, restart)
                
                try:
                    result = minimize(
                        self._objective_misfit,
                        x0,
                        method='SLSQP',
                        bounds=box_bounds,
                        constraints=constraints,
                        options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
                    )
                    if result.fun < best_fun:
                        best_fun = result.fun
                        best_result = result
                except Exception:
                    pass
        
        elif method == 'differential_evolution':
            # DE with NonlinearConstraint - no log barrier needed
            result = differential_evolution(
                self._objective_misfit,
                box_bounds,
                constraints=disk_constraint,
                maxiter=max(2000, maxiter),
                seed=42,
                polish=True,
                workers=1,
                mutation=(0.5, 1.0),
                recombination=0.7
            )
            best_result = result
            
        elif method == 'trust-constr':
            # trust-constr with NonlinearConstraint
            init_strategies = ['spread', 'circle'] + ['random'] * max(0, n_restarts - 2)
            
            for restart, init_type in enumerate(init_strategies[:n_restarts]):
                x0 = self._get_initial_guess(init_type, restart)
                result = minimize(
                    self._objective_misfit,
                    x0,
                    method='trust-constr',
                    bounds=box_bounds,
                    constraints=disk_constraint,
                    options={'maxiter': maxiter}
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            
        elif method == 'basinhopping':
            from scipy.optimize import basinhopping
            x0 = self._get_initial_guess('spread', 0)
            
            # Basinhopping with L-BFGS-B local optimizer using log barrier
            bounds_barrier = [(None, None)] * (2*n) + [(-5.0, 5.0)] * n
            minimizer_kwargs = {
                'method': 'L-BFGS-B', 
                'bounds': bounds_barrier,
                'args': (mu,)
            }
            result = basinhopping(
                self._objective_with_barrier, 
                x0, 
                minimizer_kwargs=minimizer_kwargs,
                niter=maxiter, 
                seed=42
            )
            best_result = result
            
        else:  # L-BFGS-B (default) or other gradient-based
            # L-BFGS-B with log barrier - positions UNCONSTRAINED
            bounds_barrier = [(None, None)] * (2*n) + [(-5.0, 5.0)] * n
            
            init_strategies = ['spread', 'circle'] + ['random'] * max(0, n_restarts - 2)
            
            for restart, init_type in enumerate(init_strategies[:n_restarts]):
                x0 = self._get_initial_guess(init_type, restart)
                result = minimize(
                    self._objective_with_barrier, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=bounds_barrier,
                    args=(mu,),
                    options={'maxiter': maxiter}
                )
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
