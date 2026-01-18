"""
IPOPT-Based Nonlinear Inverse Solver
=====================================

This module implements nonlinear inverse solvers using IPOPT (Interior Point OPTimizer)
via the cyipopt Python interface. IPOPT is chosen to match MATLAB's fmincon behavior,
which uses an interior-point algorithm.

Key Design Decisions
--------------------
1. **Cartesian Parameterization**: Use (x, y) coordinates, NOT polar
   - Avoids singularity at r=0
   - Better numerical stability at domain edges
   
2. **Intensity Centering**: Enforce sum=0 via q_centered = q - mean(q)
   - All n intensities are optimization variables
   - Symmetric treatment avoids edge instabilities from n-1 parameterization
   - Mathematically equivalent to explicit equality constraint
   
3. **Constraint Handling**:
   - Disk constraint: x² + y² ≤ r_max² as nonlinear inequality
   - Sum=0: Handled implicitly via centering (more stable than Aeq·x = beq)

IPOPT Settings (matching MATLAB fmincon)
----------------------------------------
- hessian_approximation = 'limited-memory' (L-BFGS behavior)
- max_iter = 30000
- tol = 1e-16

Installation
------------
cyipopt requires IPOPT library. Install via conda:
    conda install -c conda-forge cyipopt

Usage
-----
>>> solver = IPOPTNonlinearInverseSolver(n_sources=4, n_boundary=100)
>>> solver.set_measured_data(u_measured)
>>> result = solver.solve(n_restarts=10)

Author: Claude (Anthropic)
Date: January 2026
Version: 1.0.0
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings

# Check for cyipopt availability
try:
    import cyipopt
    HAS_CYIPOPT = True
except ImportError:
    HAS_CYIPOPT = False
    warnings.warn(
        "cyipopt not available. Install via: conda install -c conda-forge cyipopt\n"
        "IPOPTNonlinearInverseSolver will raise ImportError when instantiated."
    )

# Import from package
try:
    from .analytical_solver import (
        AnalyticalForwardSolver,
        greens_function_disk_neumann,
        greens_function_disk_neumann_gradient,
        Source,
        InverseResult
    )
except ImportError:
    from analytical_solver import (
        AnalyticalForwardSolver,
        greens_function_disk_neumann,
        greens_function_disk_neumann_gradient,
        Source,
        InverseResult
    )


# =============================================================================
# IPOPT PROBLEM CLASS FOR DISK DOMAIN
# =============================================================================

class IPOPTDiskProblem:
    """
    IPOPT problem formulation for inverse source localization on unit disk.
    
    This class implements the interface required by cyipopt:
    - objective(x): Returns scalar objective value
    - gradient(x): Returns gradient array
    - constraints(x): Returns constraint values
    - jacobian(x): Returns Jacobian of constraints
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    forward_solver : AnalyticalForwardSolver
        Forward solver instance
    u_measured : np.ndarray
        Measured boundary potential (mean-centered)
    r_max : float
        Maximum radius for sources (default: 0.95)
    use_analytical_gradient : bool
        Whether to use analytical gradient (default: True)
    """
    
    def __init__(self, n_sources: int, forward_solver: AnalyticalForwardSolver,
                 u_measured: np.ndarray, r_max: float = 0.95,
                 use_analytical_gradient: bool = True):
        self.n_sources = n_sources
        self.forward = forward_solver
        self.u_measured = u_measured - np.mean(u_measured)
        self.r_max = r_max
        self.use_analytical_gradient = use_analytical_gradient
        
        # Problem dimensions
        # Variables: [x₁, y₁, x₂, y₂, ..., xₙ, yₙ, q₁, q₂, ..., qₙ]
        self.n_vars = 3 * n_sources  # 2n positions + n intensities
        self.n_constraints = n_sources  # One disk constraint per source
        
        # Iteration counter
        self.n_evals = 0
        self.history = []
    
    def _params_to_sources(self, x: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """
        Convert optimization parameters to source list.
        
        Parameters layout: [x₁, y₁, x₂, y₂, ..., xₙ, yₙ, q₁, q₂, ..., qₙ]
        
        Intensities are centered to enforce sum=0 constraint.
        """
        n = self.n_sources
        sources = []
        
        # Extract positions
        positions = [(x[2*i], x[2*i + 1]) for i in range(n)]
        
        # Extract and center intensities
        intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)  # Enforce sum=0
        
        for i in range(n):
            sources.append((positions[i], intensities[i]))
        
        return sources
    
    def objective(self, x: np.ndarray) -> float:
        """
        Compute objective: ||u_computed - u_measured||²
        
        This is a pure misfit function - constraints handled separately by IPOPT.
        """
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        
        self.n_evals += 1
        self.history.append(misfit)
        
        return misfit
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective w.r.t. parameters.
        
        Uses chain rule:
        ∂f/∂xₖ = 2 Σᵢ (uᵢ - uᵢᵐᵉᵃˢ) · ∂uᵢ/∂xₖ
        """
        if not self.use_analytical_gradient:
            # Numerical gradient via finite differences
            return self._numerical_gradient(x)
        
        n = self.n_sources
        sources = self._params_to_sources(x)
        
        # Compute forward solution
        u_computed = self.forward.solve(sources)
        residual = u_computed - self.u_measured  # Shape: (n_boundary,)
        
        # Initialize gradient
        grad = np.zeros(self.n_vars)
        
        # Get boundary points
        boundary_pts = self.forward.boundary_points  # Shape: (n_boundary, 2)
        
        # Centered intensities
        raw_intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = raw_intensities - np.mean(raw_intensities)
        
        # Gradient w.r.t. positions
        for k in range(n):
            pos_k = np.array([x[2*k], x[2*k + 1]])
            q_k = intensities[k]
            
            # ∂G/∂ξ at all boundary points
            dG_dxi = greens_function_disk_neumann_gradient(boundary_pts, pos_k)
            
            # ∂u/∂xₖ = qₖ · ∂G/∂ξₓ, ∂u/∂yₖ = qₖ · ∂G/∂ξᵧ
            du_dx = q_k * dG_dxi[:, 0]
            du_dy = q_k * dG_dxi[:, 1]
            
            # Chain rule: ∂f/∂xₖ = 2 Σᵢ residualᵢ · ∂uᵢ/∂xₖ
            grad[2*k] = 2 * np.dot(residual, du_dx)
            grad[2*k + 1] = 2 * np.dot(residual, du_dy)
        
        # Gradient w.r.t. intensities (with centering correction)
        # u = Σ qₖ_centered · G(·, ξₖ) where qₖ_centered = qₖ - mean(q)
        # ∂u/∂qⱼ = ∂qⱼ_centered/∂qⱼ · G(·, ξⱼ) + Σₖ (∂qₖ_centered/∂qⱼ) · G(·, ξₖ)
        # ∂qₖ_centered/∂qⱼ = δₖⱼ - 1/n
        # So ∂u/∂qⱼ = G(·, ξⱼ) - (1/n) Σₖ G(·, ξₖ)
        
        G_all = np.zeros((len(boundary_pts), n))
        for k in range(n):
            pos_k = np.array([x[2*k], x[2*k + 1]])
            G_all[:, k] = greens_function_disk_neumann(boundary_pts, pos_k)
        
        G_mean = np.mean(G_all, axis=1)
        
        for j in range(n):
            du_dqj = G_all[:, j] - G_mean
            grad[2*n + j] = 2 * np.dot(residual, du_dqj)
        
        return grad
    
    def _numerical_gradient(self, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Finite difference gradient for validation."""
        grad = np.zeros(self.n_vars)
        f0 = self.objective(x)
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.objective(x_plus) - f0) / eps
        
        return grad
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint values: r_max² - (xₖ² + yₖ²) for each source.
        
        Constraint satisfied when value ≥ 0 (source inside disk of radius r_max).
        """
        n = self.n_sources
        c = np.zeros(n)
        
        for k in range(n):
            xk, yk = x[2*k], x[2*k + 1]
            c[k] = self.r_max**2 - (xk**2 + yk**2)
        
        return c
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of constraints.
        
        J[k, 2k] = -2xₖ, J[k, 2k+1] = -2yₖ, all other entries are 0.
        
        Returns ONLY the non-zero values matching jacobianstructure order.
        """
        n = self.n_sources
        
        # Return only non-zero values in same order as jacobianstructure
        # Structure is: [(0, 0), (0, 1), (1, 2), (1, 3), ...]
        # Values are: [-2x₀, -2y₀, -2x₁, -2y₁, ...]
        jac_values = []
        
        for k in range(n):
            jac_values.append(-2 * x[2*k])      # ∂cₖ/∂xₖ
            jac_values.append(-2 * x[2*k + 1])  # ∂cₖ/∂yₖ
        
        return np.array(jac_values)
    
    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sparsity structure of Jacobian.
        
        Returns (row_indices, col_indices) for non-zero entries.
        """
        n = self.n_sources
        rows = []
        cols = []
        
        for k in range(n):
            rows.extend([k, k])
            cols.extend([2*k, 2*k + 1])
        
        return (np.array(rows), np.array(cols))


# =============================================================================
# IPOPT NONLINEAR INVERSE SOLVER
# =============================================================================

class IPOPTNonlinearInverseSolver:
    """
    Nonlinear inverse solver using IPOPT optimizer.
    
    This is the recommended solver for achieving MATLAB-equivalent results.
    Uses interior-point optimization with proper constraint handling.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary measurement points
    sensor_locations : np.ndarray, optional
        Custom sensor locations. If None, uses evenly spaced points on unit circle.
    r_max : float
        Maximum source radius (default: 0.95)
    r_min : float
        Minimum source radius for initialization (default: 0.3)
    S_max : float
        Maximum absolute intensity (default: 5.0)
    
    Example
    -------
    >>> # Generate synthetic data
    >>> sources_true = [((-0.5, 0.3), 1.0), ((0.4, -0.4), -1.0)]
    >>> forward = AnalyticalForwardSolver(100)
    >>> u_measured = forward.solve(sources_true)
    >>> 
    >>> # Solve inverse problem
    >>> solver = IPOPTNonlinearInverseSolver(n_sources=2, n_boundary=100)
    >>> solver.set_measured_data(u_measured)
    >>> result = solver.solve(n_restarts=10)
    >>> print(f"Position RMSE: {result.residual:.2e}")
    """
    
    def __init__(self, n_sources: int, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None,
                 r_max: float = 0.95, r_min: float = 0.3, S_max: float = 5.0):
        if not HAS_CYIPOPT:
            raise ImportError(
                "cyipopt is required for IPOPTNonlinearInverseSolver.\n"
                "Install via: conda install -c conda-forge cyipopt\n"
                "If running in cloud environment without cyipopt, please run locally."
            )
        
        self.n_sources = n_sources
        self.n_boundary = n_boundary
        self.r_max = r_max
        self.r_min = r_min
        self.S_max = S_max
        
        # Create forward solver
        self.forward = AnalyticalForwardSolver(n_boundary, sensor_locations=sensor_locations)
        
        # Storage
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _generate_initial_guess(self, seed: int) -> np.ndarray:
        """
        Generate initial guess with sources inside disk.
        
        Uses well-separated angular positions with random radii.
        """
        n = self.n_sources
        np.random.seed(seed)
        
        x0 = []
        
        # Positions: evenly spaced angles with perturbation, random radii
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        angles = base_angles + np.random.uniform(-0.3, 0.3, n)
        radii = np.random.uniform(self.r_min, self.r_max * 0.9, n)
        
        for i in range(n):
            x0.append(radii[i] * np.cos(angles[i]))
            x0.append(radii[i] * np.sin(angles[i]))
        
        # Intensities: alternating signs with small random perturbation
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            x0.append(sign * (1.0 + 0.2 * np.random.randn()))
        
        return np.array(x0)
    
    def solve(self, n_restarts: int = 10, max_iter: int = 30000, tol: float = 1e-12,
              verbose: bool = False, print_level: int = 0) -> InverseResult:
        """
        Solve the nonlinear inverse problem using IPOPT.
        
        Parameters
        ----------
        n_restarts : int
            Number of random restarts (default: 10)
        max_iter : int
            Maximum iterations per restart (default: 30000)
        tol : float
            Convergence tolerance (default: 1e-12)
        verbose : bool
            Whether to print progress (default: False)
        print_level : int
            IPOPT print level (0-12, default: 0 for silent)
        
        Returns
        -------
        result : InverseResult
            Recovered sources, residual, and optimization info
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
        
        n = self.n_sources
        
        # Variable bounds
        # Positions: within bounding box of disk
        # Intensities: bounded
        lb = []
        ub = []
        for _ in range(n):
            lb.extend([-self.r_max, -self.r_max])  # x, y lower bounds
            ub.extend([self.r_max, self.r_max])     # x, y upper bounds
        for _ in range(n):
            lb.append(-self.S_max)  # intensity lower bound
            ub.append(self.S_max)   # intensity upper bound
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Constraint bounds: cₖ = r_max² - (xₖ² + yₖ²) ≥ 0
        cl = np.zeros(n)  # Lower bound: 0
        cu = np.full(n, np.inf)  # Upper bound: infinity (no upper limit)
        
        best_result = None
        best_fun = np.inf
        all_history = []
        
        for restart in range(n_restarts):
            if verbose:
                print(f"Restart {restart + 1}/{n_restarts}...", end=" ")
            
            # Create problem instance
            problem = IPOPTDiskProblem(
                n_sources=n,
                forward_solver=self.forward,
                u_measured=self.u_measured,
                r_max=self.r_max
            )
            
            # Create IPOPT problem
            nlp = cyipopt.Problem(
                n=problem.n_vars,
                m=problem.n_constraints,
                problem_obj=problem,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
            )
            
            # Set IPOPT options (matching MATLAB fmincon)
            nlp.add_option('hessian_approximation', 'limited-memory')
            nlp.add_option('max_iter', max_iter)
            nlp.add_option('tol', tol)
            nlp.add_option('acceptable_tol', tol * 100)
            nlp.add_option('print_level', print_level)
            nlp.add_option('sb', 'yes')  # Suppress banner
            
            # Additional options for better convergence
            nlp.add_option('mu_strategy', 'adaptive')
            nlp.add_option('nlp_scaling_method', 'gradient-based')
            
            # Initial guess
            x0 = self._generate_initial_guess(seed=42 + restart)
            
            # Solve
            try:
                x_opt, info = nlp.solve(x0)
                
                # Evaluate final objective
                final_obj = problem.objective(x_opt)
                
                if verbose:
                    print(f"obj = {final_obj:.2e}, status = {info['status_msg']}")
                
                if final_obj < best_fun:
                    best_fun = final_obj
                    best_result = {
                        'x': x_opt,
                        'fun': final_obj,
                        'info': info,
                        'history': problem.history.copy()
                    }
                
                all_history.extend(problem.history)
                
            except Exception as e:
                if verbose:
                    print(f"failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All IPOPT restarts failed")
        
        # Extract sources from best result
        x_opt = best_result['x']
        sources = []
        
        # Positions
        positions = [(x_opt[2*i], x_opt[2*i + 1]) for i in range(n)]
        
        # Centered intensities
        raw_intensities = np.array([x_opt[2*n + i] for i in range(n)])
        intensities = raw_intensities - np.mean(raw_intensities)
        
        for i in range(n):
            sources.append(Source(
                x=positions[i][0],
                y=positions[i][1],
                intensity=intensities[i]
            ))
        
        self.history = all_history
        
        return InverseResult(
            sources=sources,
            residual=np.sqrt(best_result['fun']),
            success=best_result['info']['status'] == 0,
            message=best_result['info']['status_msg'],
            iterations=len(all_history),
            history=all_history
        )


# =============================================================================
# IPOPT PROBLEM CLASS FOR CONFORMAL DOMAINS
# =============================================================================

class IPOPTConformalProblem:
    """
    IPOPT problem formulation for inverse source localization on conformal domains.
    
    Uses conformal mapping to transform arbitrary simply-connected domains to unit disk.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    conformal_map : ConformalMap
        Conformal map instance (from conformal_solver module)
    forward_solver : ConformalForwardSolver
        Forward solver instance
    u_measured : np.ndarray
        Measured boundary potential (mean-centered)
    """
    
    def __init__(self, n_sources: int, conformal_map, forward_solver,
                 u_measured: np.ndarray):
        self.n_sources = n_sources
        self.map = conformal_map
        self.forward = forward_solver
        self.u_measured = u_measured - np.mean(u_measured)
        
        # Problem dimensions
        self.n_vars = 3 * n_sources
        self.n_constraints = n_sources
        
        # Get domain bounds from conformal map
        boundary = self.map.boundary_physical(100)
        self.x_min, self.x_max = np.real(boundary).min(), np.real(boundary).max()
        self.y_min, self.y_max = np.imag(boundary).min(), np.imag(boundary).max()
        
        # Iteration counter
        self.n_evals = 0
        self.history = []
    
    def _params_to_sources(self, x: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Convert parameters to sources with centered intensities."""
        n = self.n_sources
        sources = []
        
        positions = [(x[2*i], x[2*i + 1]) for i in range(n)]
        intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)
        
        for i in range(n):
            sources.append((positions[i], intensities[i]))
        
        return sources
    
    def objective(self, x: np.ndarray) -> float:
        """Compute misfit objective."""
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        
        self.n_evals += 1
        self.history.append(misfit)
        
        return misfit
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Numerical gradient (conformal mapping makes analytical gradient complex)."""
        eps = 1e-7
        grad = np.zeros(self.n_vars)
        f0 = self.objective(x)
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (self.objective(x_plus) - f0) / eps
        
        return grad
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Conformal domain constraint: 1 - |f(zₖ)|² ≥ 0 for each source.
        
        Source is inside domain iff its image under conformal map is inside unit disk.
        """
        n = self.n_sources
        c = np.zeros(n)
        
        for k in range(n):
            z = complex(x[2*k], x[2*k + 1])
            w = self.map.to_disk(np.array([z]))[0]
            c[k] = 1.0 - np.abs(w)**2
        
        return c
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Numerical Jacobian of constraints."""
        eps = 1e-7
        c0 = self.constraints(x)
        n = self.n_sources
        
        jac = np.zeros((n, self.n_vars))
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            jac[:, i] = (self.constraints(x_plus) - c0) / eps
        
        return jac.flatten()


class IPOPTConformalInverseSolver:
    """
    IPOPT-based nonlinear inverse solver for general conformal domains.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    conformal_map : ConformalMap
        Conformal map from domain to unit disk
    n_boundary : int
        Number of boundary measurement points
    S_max : float
        Maximum absolute intensity (default: 5.0)
    """
    
    def __init__(self, n_sources: int, conformal_map, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None, S_max: float = 5.0):
        if not HAS_CYIPOPT:
            raise ImportError(
                "cyipopt is required for IPOPTConformalInverseSolver.\n"
                "Install via: conda install -c conda-forge cyipopt"
            )
        
        # Import here to avoid circular dependency
        try:
            from .conformal_solver import ConformalForwardSolver
        except ImportError:
            from conformal_solver import ConformalForwardSolver
        
        self.n_sources = n_sources
        self.map = conformal_map
        self.S_max = S_max
        
        # Create forward solver
        self.forward = ConformalForwardSolver(conformal_map, n_boundary,
                                               sensor_locations=sensor_locations)
        
        # Get domain bounds
        boundary = conformal_map.boundary_physical(100)
        self.x_min, self.x_max = np.real(boundary).min(), np.real(boundary).max()
        self.y_min, self.y_max = np.imag(boundary).min(), np.imag(boundary).max()
        
        # Storage
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _generate_initial_guess(self, seed: int) -> np.ndarray:
        """Generate initial guess with sources inside domain."""
        n = self.n_sources
        np.random.seed(seed)
        
        x0 = []
        
        # Use rejection sampling to get points inside domain
        max_attempts = 1000
        
        for i in range(n):
            for attempt in range(max_attempts):
                x = np.random.uniform(self.x_min, self.x_max)
                y = np.random.uniform(self.y_min, self.y_max)
                z = complex(x, y)
                
                if self.map.is_inside(z):
                    # Additional check: not too close to boundary
                    w = self.map.to_disk(np.array([z]))[0]
                    if np.abs(w) < 0.9:
                        x0.extend([x, y])
                        break
            else:
                # Fallback: use centroid
                boundary = self.map.boundary_physical(100)
                cx = np.real(boundary).mean()
                cy = np.imag(boundary).mean()
                x0.extend([cx + 0.1 * np.random.randn(),
                          cy + 0.1 * np.random.randn()])
        
        # Intensities
        for i in range(n):
            sign = 1.0 if i % 2 == 0 else -1.0
            x0.append(sign * (1.0 + 0.2 * np.random.randn()))
        
        return np.array(x0)
    
    def solve(self, n_restarts: int = 10, max_iter: int = 30000, tol: float = 1e-12,
              verbose: bool = False, print_level: int = 0) -> InverseResult:
        """
        Solve the nonlinear inverse problem using IPOPT.
        
        Parameters
        ----------
        n_restarts : int
            Number of random restarts
        max_iter : int
            Maximum iterations per restart
        tol : float
            Convergence tolerance
        verbose : bool
            Whether to print progress
        print_level : int
            IPOPT print level
        
        Returns
        -------
        result : InverseResult
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
        
        n = self.n_sources
        
        # Variable bounds (box containing domain)
        lb = []
        ub = []
        margin = 0.1 * max(self.x_max - self.x_min, self.y_max - self.y_min)
        
        for _ in range(n):
            lb.extend([self.x_min - margin, self.y_min - margin])
            ub.extend([self.x_max + margin, self.y_max + margin])
        for _ in range(n):
            lb.append(-self.S_max)
            ub.append(self.S_max)
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Constraint bounds
        cl = np.zeros(n)
        cu = np.full(n, np.inf)
        
        best_result = None
        best_fun = np.inf
        all_history = []
        
        for restart in range(n_restarts):
            if verbose:
                print(f"Restart {restart + 1}/{n_restarts}...", end=" ")
            
            problem = IPOPTConformalProblem(
                n_sources=n,
                conformal_map=self.map,
                forward_solver=self.forward,
                u_measured=self.u_measured
            )
            
            nlp = cyipopt.Problem(
                n=problem.n_vars,
                m=problem.n_constraints,
                problem_obj=problem,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
            )
            
            nlp.add_option('hessian_approximation', 'limited-memory')
            nlp.add_option('max_iter', max_iter)
            nlp.add_option('tol', tol)
            nlp.add_option('acceptable_tol', tol * 100)
            nlp.add_option('print_level', print_level)
            nlp.add_option('sb', 'yes')
            nlp.add_option('mu_strategy', 'adaptive')
            
            x0 = self._generate_initial_guess(seed=42 + restart)
            
            try:
                x_opt, info = nlp.solve(x0)
                final_obj = problem.objective(x_opt)
                
                if verbose:
                    print(f"obj = {final_obj:.2e}, status = {info['status_msg']}")
                
                if final_obj < best_fun:
                    best_fun = final_obj
                    best_result = {
                        'x': x_opt,
                        'fun': final_obj,
                        'info': info,
                        'history': problem.history.copy()
                    }
                
                all_history.extend(problem.history)
                
            except Exception as e:
                if verbose:
                    print(f"failed: {e}")
                continue
        
        if best_result is None:
            raise RuntimeError("All IPOPT restarts failed")
        
        x_opt = best_result['x']
        sources = []
        
        positions = [(x_opt[2*i], x_opt[2*i + 1]) for i in range(n)]
        raw_intensities = np.array([x_opt[2*n + i] for i in range(n)])
        intensities = raw_intensities - np.mean(raw_intensities)
        
        for i in range(n):
            sources.append(Source(
                x=positions[i][0],
                y=positions[i][1],
                intensity=intensities[i]
            ))
        
        self.history = all_history
        
        return InverseResult(
            sources=sources,
            residual=np.sqrt(best_result['fun']),
            success=best_result['info']['status'] == 0,
            message=best_result['info']['status_msg'],
            iterations=len(all_history),
            history=all_history
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_cyipopt_available() -> bool:
    """Check if cyipopt is available."""
    return HAS_CYIPOPT


def get_ipopt_version() -> Optional[str]:
    """Get IPOPT version if available."""
    if not HAS_CYIPOPT:
        return None
    try:
        return cyipopt.__version__
    except AttributeError:
        return "unknown"


# =============================================================================
# TEST SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("IPOPT Nonlinear Inverse Solver Test")
    print("=" * 70)
    
    if not HAS_CYIPOPT:
        print("\nERROR: cyipopt not available.")
        print("Install via: conda install -c conda-forge cyipopt")
        exit(1)
    
    print(f"\ncyipopt version: {get_ipopt_version()}")
    
    # Test configuration
    n_sources = 4
    n_boundary = 100
    noise_level = 0.0
    
    # Create well-separated test sources
    print(f"\nCreating {n_sources} well-separated test sources...")
    sources_true = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += 0.1 * np.random.randn(n_sources)  # Small perturbation
    
    for i, theta in enumerate(angles):
        r = 0.6 + 0.2 * np.random.rand()  # r in [0.6, 0.8]
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        intensity = 1.0 if i % 2 == 0 else -1.0
        sources_true.append(((x, y), intensity))
    
    # Enforce zero sum
    total = sum(s[1] for s in sources_true)
    sources_true[-1] = (sources_true[-1][0], sources_true[-1][1] - total)
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    # Generate measurement data
    print("\nGenerating measurement data...")
    forward = AnalyticalForwardSolver(n_boundary)
    u_measured = forward.solve(sources_true)
    
    if noise_level > 0:
        np.random.seed(42)
        u_measured += noise_level * np.random.randn(len(u_measured))
        print(f"Added noise: σ = {noise_level}")
    
    # Solve inverse problem
    print("\nSolving inverse problem with IPOPT...")
    solver = IPOPTNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_boundary)
    solver.set_measured_data(u_measured)
    
    result = solver.solve(n_restarts=10, verbose=True)
    
    # Print results
    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    
    print("\nRecovered sources:")
    for i, s in enumerate(result.sources):
        print(f"  {i+1}: ({s.x:+.4f}, {s.y:+.4f}), q={s.intensity:+.4f}")
    
    # Compute position error
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    cost = np.zeros((n, n))
    for i, ((tx, ty), _) in enumerate(sources_true):
        for j, s in enumerate(result.sources):
            cost[i, j] = np.sqrt((tx - s.x)**2 + (ty - s.y)**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    pos_errors = cost[row_ind, col_ind]
    
    print(f"\nPosition errors:")
    for i, (ri, ci) in enumerate(zip(row_ind, col_ind)):
        print(f"  Source {ri+1} -> Recovered {ci+1}: {pos_errors[i]:.2e}")
    
    print(f"\nMean position error: {pos_errors.mean():.2e}")
    print(f"Max position error:  {pos_errors.max():.2e}")
    print(f"Residual (RMS):      {result.residual:.2e}")
    print(f"Success:             {result.success}")
    print(f"Message:             {result.message}")
    
    # Check success criteria
    print("\n" + "=" * 70)
    if pos_errors.mean() < 1e-5:
        print("✓ SUCCESS: Position RMSE < 1e-5 (target achieved)")
    elif pos_errors.mean() < 1e-3:
        print("~ ACCEPTABLE: Position RMSE < 1e-3")
    else:
        print("✗ NEEDS WORK: Position RMSE > 1e-3")
    print("=" * 70)
