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
   - Conformal domain constraint: 1 - |f(z)|² ≥ 0
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

DO NOT use pip install - it requires system IPOPT library.

Usage
-----
>>> solver = IPOPTNonlinearInverseSolver(n_sources=4, n_boundary=100)
>>> solver.set_measured_data(u_measured)
>>> result = solver.solve(n_restarts=10)

Version: 1.0.0
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Check for cyipopt availability - NO FALLBACK
try:
    import cyipopt
    HAS_CYIPOPT = True
except ImportError:
    HAS_CYIPOPT = False


# Import from package
try:
    from .analytical_solver import (
        AnalyticalForwardSolver,
        greens_function_disk_neumann,
        Source,
        InverseResult
    )
except ImportError:
    from analytical_solver import (
        AnalyticalForwardSolver,
        greens_function_disk_neumann,
        Source,
        InverseResult
    )


def _check_cyipopt():
    """Check cyipopt availability and raise clear error if not found."""
    if not HAS_CYIPOPT:
        raise ImportError(
            "cyipopt is required but not installed.\n"
            "Install via conda (NOT pip):\n"
            "    conda install -c conda-forge cyipopt\n\n"
            "If running in a cloud environment without conda, "
            "please run this code locally where cyipopt is installed."
        )


# =============================================================================
# IPOPT PROBLEM CLASS FOR DISK DOMAIN
# =============================================================================

class IPOPTDiskProblem:
    """
    IPOPT problem formulation for inverse source localization on unit disk.
    
    Implements the interface required by cyipopt:
    - objective(x): Returns scalar objective value
    - gradient(x): Returns gradient array
    - constraints(x): Returns constraint values
    - jacobian(x): Returns Jacobian of constraints (sparse values)
    - jacobianstructure(): Returns sparsity pattern of Jacobian
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    forward_solver : AnalyticalForwardSolver
        Forward solver instance
    u_measured : np.ndarray
        Measured boundary potential (will be mean-centered)
    """
    
    def __init__(self, n_sources: int, forward_solver: AnalyticalForwardSolver,
                 u_measured: np.ndarray):
        self.n_sources = n_sources
        self.forward = forward_solver
        self.u_measured = u_measured - np.mean(u_measured)
        
        # Problem dimensions
        # Variables: [x₁, y₁, x₂, y₂, ..., xₙ, yₙ, q₁, q₂, ..., qₙ]
        self.n_vars = 3 * n_sources  # 2n positions + n intensities
        self.n_constraints = n_sources  # One disk constraint per source
        
        # Iteration counter for diagnostics
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
        
        # Extract and center intensities (enforces sum=0)
        intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)
        
        for i in range(n):
            sources.append((positions[i], intensities[i]))
        
        return sources
    
    def objective(self, x: np.ndarray) -> float:
        """
        Compute objective: ||u_computed - u_measured||²
        
        Note: The forward solver evaluates for ANY source position.
        IPOPT constraints ensure we converge to feasible solutions.
        """
        # Store current x for intermediate callback access
        self._last_x = x.copy()
        
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        
        self.n_evals += 1
        self.history.append(misfit)
        
        return misfit
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                    mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        IPOPT intermediate callback - called after each iteration.
        
        Used for visualization updates. Returns True to continue optimization.
        """
        if hasattr(self, '_visualizer') and self._visualizer is not None:
            if hasattr(self, '_last_x'):
                self._visualizer.update(iter_count, obj_value, self._last_x)
        return True
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of objective via finite differences.
        """
        eps = 1e-7
        grad = np.zeros(self.n_vars)
        
        # Compute objective directly without recording to history
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        f0 = np.sum((u_computed - self.u_measured)**2)
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            sources_plus = self._params_to_sources(x_plus)
            u_plus = self.forward.solve(sources_plus)
            f_plus = np.sum((u_plus - self.u_measured)**2)
            grad[i] = (f_plus - f0) / eps
        
        return grad
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint values: 1 - (xₖ² + yₖ²) for each source.
        
        Constraint satisfied when value >= 0 (source strictly inside unit disk).
        """
        n = self.n_sources
        c = np.zeros(n)
        
        for k in range(n):
            xk, yk = x[2*k], x[2*k + 1]
            c[k] = 1.0 - (xk**2 + yk**2)
        
        return c
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of constraints.
        
        For c[k] = 1 - xₖ² - yₖ²:
        ∂cₖ/∂xₖ = -2xₖ, ∂cₖ/∂yₖ = -2yₖ, all other entries are 0.
        
        Returns ONLY the non-zero values matching jacobianstructure order.
        """
        n = self.n_sources
        jac_values = []
        
        for k in range(n):
            jac_values.append(-2 * x[2*k])      # ∂cₖ/∂xₖ
            jac_values.append(-2 * x[2*k + 1])  # ∂cₖ/∂yₖ
        
        return np.array(jac_values)
    
    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return sparsity structure of constraint Jacobian.
        
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
# IPOPT NONLINEAR INVERSE SOLVER FOR DISK
# =============================================================================

class IPOPTNonlinearInverseSolver:
    """
    Nonlinear inverse solver using IPOPT optimizer for unit disk domain.
    
    This is the recommended solver for achieving MATLAB-equivalent results
    on the unit disk. Uses interior-point optimization with proper nonlinear
    constraint handling.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary measurement points
    sensor_locations : np.ndarray, optional
        Custom sensor locations. If None, uses evenly spaced points on unit circle.
    S_max : float
        Maximum absolute intensity bound (default: 5.0)
    
    Example
    -------
    >>> # Generate synthetic data
    >>> sources_true = [((0.5, 0.3), 1.0), ((-0.4, -0.4), -1.0)]
    >>> forward = AnalyticalForwardSolver(100)
    >>> u_measured = forward.solve(sources_true)
    >>> 
    >>> # Solve inverse problem
    >>> solver = IPOPTNonlinearInverseSolver(n_sources=2, n_boundary=100)
    >>> solver.set_measured_data(u_measured)
    >>> result = solver.solve(n_restarts=10)
    >>> print(f"Residual: {result.residual:.2e}")
    """
    
    def __init__(self, n_sources: int, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None, S_max: float = 5.0):
        _check_cyipopt()
        
        self.n_sources = n_sources
        self.n_boundary = n_boundary
        self.S_max = S_max
        
        # Create forward solver with consistent sensor locations
        self.forward = AnalyticalForwardSolver(n_boundary, sensor_locations=sensor_locations)
        
        # Storage
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _generate_initial_guess(self, seed: int) -> np.ndarray:
        """
        Generate initial guess with sources inside unit disk.
        
        Uses well-separated angular positions with radii in [0.3, 0.85].
        """
        n = self.n_sources
        np.random.seed(seed)
        
        x0 = []
        
        # Positions: evenly spaced angles with perturbation, random radii in [0.3, 0.85]
        base_angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        angles = base_angles + np.random.uniform(-0.3, 0.3, n)
        radii = np.random.uniform(0.3, 0.85, n)
        
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
        # Positions: UNBOUNDED - nonlinear constraint handles the disk
        # Intensities: box bounded
        lb = []
        ub = []
        for _ in range(n):
            lb.extend([-2e19, -2e19])  # x, y effectively unbounded
            ub.extend([2e19, 2e19])
        for _ in range(n):
            lb.append(-self.S_max)  # intensity lower bound
            ub.append(self.S_max)   # intensity upper bound
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Nonlinear constraint: 1 - x² - y² >= 0 (unit disk)
        cl = np.zeros(n)       # Lower bound: 0
        cu = np.full(n, 2e19)  # Upper bound: large number
        
        best_result = None
        best_fun = np.inf
        all_history = []
        
        for restart in range(n_restarts):
            if verbose:
                print(f"Restart {restart + 1}/{n_restarts}...", end=" ")
            
            # Create fresh problem instance for each restart
            problem = IPOPTDiskProblem(
                n_sources=n,
                forward_solver=self.forward,
                u_measured=self.u_measured
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
                
                # Evaluate final objective (without adding to history)
                sources_temp = problem._params_to_sources(x_opt)
                u_temp = self.forward.solve(sources_temp)
                final_obj = np.sum((u_temp - self.u_measured)**2)
                
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
    
    def solve_with_visualization(self, sources_true: List[Tuple] = None,
                                  update_interval: int = 5,
                                  max_iter: int = 30000, tol: float = 1e-12,
                                  verbose: bool = True, print_level: int = 0) -> InverseResult:
        """
        Solve with real-time visualization of optimization progress.
        
        Shows live plots of:
        - Loss curve (log scale)
        - Current source positions vs true positions
        - Intensity bar chart
        - Convergence diagnostics
        
        Parameters
        ----------
        sources_true : list, optional
            True sources for comparison: [((x,y), q), ...]
        update_interval : int
            Update visualization every N iterations (default: 5)
        max_iter : int
            Maximum iterations (default: 30000)
        tol : float
            Convergence tolerance (default: 1e-12)
        verbose : bool
            Print progress (default: True)
        print_level : int
            IPOPT print level (default: 0)
        
        Returns
        -------
        result : InverseResult
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
        
        # Import visualization
        try:
            from .visualization.optimization_live import OptimizationVisualizer
        except ImportError:
            from visualization.optimization_live import OptimizationVisualizer
        
        n = self.n_sources
        
        # Create visualizer
        viz = OptimizationVisualizer(
            n_sources=n,
            sources_true=sources_true,
            domain='disk',
            domain_params={'r': 1.0},  # Unit disk
            update_interval=update_interval
        )
        
        # Variable bounds - UNBOUNDED for positions, box bounded for intensities
        lb = []
        ub = []
        for _ in range(n):
            lb.extend([-2e19, -2e19])  # x, y effectively unbounded
            ub.extend([2e19, 2e19])
        for _ in range(n):
            lb.append(-self.S_max)
            ub.append(self.S_max)
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Nonlinear constraint: 1 - x² - y² >= 0 (unit disk)
        cl = np.zeros(n)
        cu = np.full(n, 2e19)
        
        if verbose:
            print("Starting IPOPT optimization with visualization...")
            print(f"  Sources: {n}, Max iter: {max_iter}, Tol: {tol}")
        
        # Create problem with visualizer attached
        problem = IPOPTDiskProblem(
            n_sources=n,
            forward_solver=self.forward,
            u_measured=self.u_measured
        )
        problem._visualizer = viz
        
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
        
        # IPOPT options
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('max_iter', max_iter)
        nlp.add_option('tol', tol)
        nlp.add_option('acceptable_tol', tol * 100)
        nlp.add_option('print_level', print_level)
        nlp.add_option('sb', 'yes')
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('nlp_scaling_method', 'gradient-based')
        
        # Initial guess
        x0 = self._generate_initial_guess(seed=42)
        
        # Solve
        try:
            x_opt, info = nlp.solve(x0)
            
            # Extract sources
            positions = [(x_opt[2*i], x_opt[2*i + 1]) for i in range(n)]
            raw_intensities = np.array([x_opt[2*n + i] for i in range(n)])
            intensities = raw_intensities - np.mean(raw_intensities)
            
            sources = []
            for i in range(n):
                sources.append(Source(
                    x=positions[i][0],
                    y=positions[i][1],
                    intensity=intensities[i]
                ))
            
            result = InverseResult(
                sources=sources,
                residual=np.sqrt(problem.history[-1]) if problem.history else 0,
                success=info['status'] == 0,
                message=info['status_msg'],
                iterations=len(problem.history),
                history=problem.history
            )
            
            if verbose:
                print(f"\nOptimization complete: {info['status_msg']}")
            
        except Exception as e:
            if verbose:
                print(f"\nOptimization failed: {e}")
            viz.close()
            raise
        
        # Finalize visualization
        viz.finalize(result)
        
        return result


# =============================================================================
# IPOPT PROBLEM CLASS FOR CONFORMAL DOMAINS
# =============================================================================

class IPOPTConformalProblem:
    """
    IPOPT problem formulation for inverse source localization on conformal domains.
    
    Uses conformal mapping to transform arbitrary simply-connected domains to unit disk.
    Constraint: source must be inside domain, i.e., |f(z)| < 1 where f is conformal map.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to recover
    conformal_map : ConformalMap
        Conformal map instance (from conformal_solver module)
    forward_solver : ConformalForwardSolver
        Forward solver instance
    u_measured : np.ndarray
        Measured boundary potential (will be mean-centered)
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
        
        # Diagnostics
        self.n_evals = 0
        self.history = []
    
    def _params_to_sources(self, x: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """Convert parameters to sources with intensity centering."""
        n = self.n_sources
        sources = []
        
        positions = [(x[2*i], x[2*i + 1]) for i in range(n)]
        intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)
        
        for i in range(n):
            sources.append((positions[i], intensities[i]))
        
        return sources
    
    def objective(self, x: np.ndarray) -> float:
        """Compute objective: ||u_computed - u_measured||²"""
        # Store current x for intermediate callback access
        self._last_x = x.copy()
        
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        
        misfit = np.sum((u_computed - self.u_measured)**2)
        
        self.n_evals += 1
        self.history.append(misfit)
        
        return misfit
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                    mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """IPOPT intermediate callback for visualization."""
        if hasattr(self, '_visualizer') and self._visualizer is not None:
            if hasattr(self, '_last_x'):
                self._visualizer.update(iter_count, obj_value, self._last_x)
        return True
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient via finite differences."""
        eps = 1e-7
        grad = np.zeros(self.n_vars)
        
        # Compute objective directly without recording to history
        sources = self._params_to_sources(x)
        u_computed = self.forward.solve(sources)
        f0 = np.sum((u_computed - self.u_measured)**2)
        
        for i in range(self.n_vars):
            x_plus = x.copy()
            x_plus[i] += eps
            sources_plus = self._params_to_sources(x_plus)
            u_plus = self.forward.solve(sources_plus)
            f_plus = np.sum((u_plus - self.u_measured)**2)
            grad[i] = (f_plus - f0) / eps
        
        return grad
    
    def constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint values: 1 - |f(zₖ)|² for each source.
        
        Where f is the conformal map to unit disk.
        Constraint satisfied when value ≥ 0 (source inside domain).
        """
        n = self.n_sources
        c = np.zeros(n)
        
        for k in range(n):
            z = complex(x[2*k], x[2*k + 1])
            w = self.map.to_disk(z)
            c[k] = 1.0 - np.abs(w)**2
        
        return c
    
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of constraints via finite differences.
        
        For conformal map constraint, analytical Jacobian is complex.
        """
        eps = 1e-7
        n = self.n_sources
        jac_values = []
        
        c0 = self.constraints(x)
        
        for k in range(n):
            # ∂cₖ/∂xₖ
            x_plus = x.copy()
            x_plus[2*k] += eps
            c_plus = self.constraints(x_plus)
            jac_values.append((c_plus[k] - c0[k]) / eps)
            
            # ∂cₖ/∂yₖ
            x_plus = x.copy()
            x_plus[2*k + 1] += eps
            c_plus = self.constraints(x_plus)
            jac_values.append((c_plus[k] - c0[k]) / eps)
        
        return np.array(jac_values)
    
    def jacobianstructure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return sparsity structure of constraint Jacobian."""
        n = self.n_sources
        rows = []
        cols = []
        
        for k in range(n):
            rows.extend([k, k])
            cols.extend([2*k, 2*k + 1])
        
        return (np.array(rows), np.array(cols))


# =============================================================================
# IPOPT CONFORMAL INVERSE SOLVER
# =============================================================================

class IPOPTConformalInverseSolver:
    """
    Nonlinear inverse solver using IPOPT for conformal domains.
    
    Works with any simply-connected domain that has a conformal map to the disk.
    
    Parameters
    ----------
    conformal_map : ConformalMap
        Conformal mapping from physical domain to unit disk
    n_sources : int
        Number of sources to recover
    n_boundary : int
        Number of boundary measurement points
    sensor_locations : np.ndarray, optional
        Custom sensor locations on the boundary
    S_max : float
        Maximum absolute intensity bound (default: 5.0)
    """
    
    def __init__(self, conformal_map, n_sources: int, n_boundary: int = 100,
                 sensor_locations: np.ndarray = None, S_max: float = 5.0):
        _check_cyipopt()
        
        self.map = conformal_map
        self.n_sources = n_sources
        self.n_boundary = n_boundary
        self.S_max = S_max
        
        # Import ConformalForwardSolver
        try:
            from .conformal_solver import ConformalForwardSolver
        except ImportError:
            from conformal_solver import ConformalForwardSolver
        
        # Create forward solver with consistent sensor locations
        self.forward = ConformalForwardSolver(conformal_map, n_boundary,
                                               sensor_locations=sensor_locations)
        
        # Get domain bounding box
        boundary = self.map.boundary_physical(100)
        self.x_min, self.x_max = np.real(boundary).min(), np.real(boundary).max()
        self.y_min, self.y_max = np.imag(boundary).min(), np.imag(boundary).max()
        
        # Storage
        self.u_measured = None
        self.history = []
    
    def set_measured_data(self, u_measured: np.ndarray):
        """Set the boundary measurements to fit."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _generate_initial_guess(self, seed: int) -> np.ndarray:
        """
        Generate initial guess with sources inside the domain.
        
        Uses rejection sampling to ensure all sources are valid.
        """
        n = self.n_sources
        np.random.seed(seed)
        
        x0 = []
        max_attempts = 1000
        
        for i in range(n):
            # Generate position inside domain using rejection sampling
            for attempt in range(max_attempts):
                x = np.random.uniform(self.x_min, self.x_max)
                y = np.random.uniform(self.y_min, self.y_max)
                z = complex(x, y)
                
                if self.map.is_inside(z):
                    x0.extend([x, y])
                    break
            else:
                # Fallback: use domain centroid with small offset
                boundary = self.map.boundary_physical(100)
                cx = np.real(boundary).mean()
                cy = np.imag(boundary).mean()
                x0.extend([cx + 0.05 * np.random.randn(),
                          cy + 0.05 * np.random.randn()])
        
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
        
        # Variable bounds - box containing domain with margin
        margin = 0.1 * max(self.x_max - self.x_min, self.y_max - self.y_min)
        
        lb = []
        ub = []
        for _ in range(n):
            lb.extend([self.x_min - margin, self.y_min - margin])
            ub.extend([self.x_max + margin, self.y_max + margin])
        for _ in range(n):
            lb.append(-self.S_max)
            ub.append(self.S_max)
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        # Constraint bounds: cₖ = 1 - |f(zₖ)|² ≥ 0
        cl = np.zeros(n)
        cu = np.full(n, 1e20)
        
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
            
            # IPOPT options
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
                
                # Evaluate final objective
                sources_temp = problem._params_to_sources(x_opt)
                u_temp = self.forward.solve(sources_temp)
                final_obj = np.sum((u_temp - self.u_measured)**2)
                
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
    
    def solve_with_visualization(self, sources_true: List[Tuple] = None,
                                  update_interval: int = 5,
                                  max_iter: int = 30000, tol: float = 1e-12,
                                  verbose: bool = True, print_level: int = 0) -> InverseResult:
        """
        Solve with real-time visualization for conformal domains.
        
        Parameters
        ----------
        sources_true : list, optional
            True sources for comparison: [((x,y), q), ...]
        update_interval : int
            Update visualization every N iterations (default: 5)
        max_iter : int
            Maximum iterations (default: 30000)
        tol : float
            Convergence tolerance (default: 1e-12)
        verbose : bool
            Print progress (default: True)
        print_level : int
            IPOPT print level (default: 0)
        
        Returns
        -------
        result : InverseResult
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data() first")
        
        # Import visualization
        try:
            from .visualization.optimization_live import OptimizationVisualizer
        except ImportError:
            from visualization.optimization_live import OptimizationVisualizer
        
        n = self.n_sources
        
        # Determine domain type for visualization
        domain_type = 'disk'  # Default visualization as disk
        domain_params = {'r': 1.0}
        
        # Create visualizer
        viz = OptimizationVisualizer(
            n_sources=n,
            sources_true=sources_true,
            domain=domain_type,
            domain_params=domain_params,
            update_interval=update_interval
        )
        
        # Variable bounds
        margin = 0.1 * max(self.x_max - self.x_min, self.y_max - self.y_min)
        
        lb = []
        ub = []
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
        cu = np.full(n, 1e20)
        
        if verbose:
            print("Starting IPOPT optimization with visualization...")
            print(f"  Sources: {n}, Max iter: {max_iter}, Tol: {tol}")
        
        # Create problem with visualizer attached
        problem = IPOPTConformalProblem(
            n_sources=n,
            conformal_map=self.map,
            forward_solver=self.forward,
            u_measured=self.u_measured
        )
        problem._visualizer = viz
        
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
        
        # IPOPT options
        nlp.add_option('hessian_approximation', 'limited-memory')
        nlp.add_option('max_iter', max_iter)
        nlp.add_option('tol', tol)
        nlp.add_option('acceptable_tol', tol * 100)
        nlp.add_option('print_level', print_level)
        nlp.add_option('sb', 'yes')
        nlp.add_option('mu_strategy', 'adaptive')
        
        # Initial guess
        x0 = self._generate_initial_guess(seed=42)
        
        # Solve
        try:
            x_opt, info = nlp.solve(x0)
            
            # Extract sources
            positions = [(x_opt[2*i], x_opt[2*i + 1]) for i in range(n)]
            raw_intensities = np.array([x_opt[2*n + i] for i in range(n)])
            intensities = raw_intensities - np.mean(raw_intensities)
            
            sources = []
            for i in range(n):
                sources.append(Source(
                    x=positions[i][0],
                    y=positions[i][1],
                    intensity=intensities[i]
                ))
            
            result = InverseResult(
                sources=sources,
                residual=np.sqrt(problem.history[-1]) if problem.history else 0,
                success=info['status'] == 0,
                message=info['status_msg'],
                iterations=len(problem.history),
                history=problem.history
            )
            
            if verbose:
                print(f"\nOptimization complete: {info['status_msg']}")
            
        except Exception as e:
            if verbose:
                print(f"\nOptimization failed: {e}")
            viz.close()
            raise
        
        # Finalize visualization
        viz.finalize(result)
        
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_cyipopt_available() -> bool:
    """Check if cyipopt is available."""
    return HAS_CYIPOPT


def get_ipopt_version() -> Optional[str]:
    """Get IPOPT version string if available."""
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
        print("\nThis test must be run locally where cyipopt is installed.")
        exit(1)
    
    print(f"\ncyipopt version: {get_ipopt_version()}")
    
    # Test configuration
    n_sources = 4
    n_boundary = 100
    noise_level = 0.0
    
    # Create well-separated test sources
    print(f"\nCreating {n_sources} well-separated test sources...")
    np.random.seed(42)
    
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
    
    # Compute position error using optimal assignment
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
        print(f"  True {ri+1} -> Recovered {ci+1}: {pos_errors[i]:.2e}")
    
    print(f"\nMean position error: {pos_errors.mean():.2e}")
    print(f"Max position error:  {pos_errors.max():.2e}")
    print(f"Residual (RMS):      {result.residual:.2e}")
    print(f"Success:             {result.success}")
    print(f"Message:             {result.message}")
    
    # Check success criteria
    print("\n" + "=" * 70)
    if pos_errors.mean() < 1e-5:
        print("✓ SUCCESS: Mean position error < 1e-5 (target achieved)")
    elif pos_errors.mean() < 1e-3:
        print("~ ACCEPTABLE: Mean position error < 1e-3")
    else:
        print("✗ NEEDS WORK: Mean position error > 1e-3")
    print("=" * 70)
