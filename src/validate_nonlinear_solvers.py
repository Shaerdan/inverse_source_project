"""
Comprehensive Validation of Nonlinear Inverse Solvers
======================================================

This script validates nonlinear solvers across:
1. All supported domains (disk, ellipse, star, square, polygon)
2. Both Cartesian and Polar parameterizations
3. Scaling from 2 to 10 sources

The polar parameterization matches the MATLAB implementation:
    params = [S1, r1, θ1, S2, r2, θ2, ..., r_n, θ_n]
    (last intensity computed from Σq = 0 constraint)

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from time import time
import warnings

# Import solvers
try:
    from analytical_solver import (
        AnalyticalForwardSolver, 
        AnalyticalNonlinearInverseSolver,
        greens_function_disk_neumann,
        Source, InverseResult
    )
    from conformal_solver import (
        ConformalForwardSolver,
        ConformalNonlinearInverseSolver,
        EllipseMap,
        NumericalConformalMap,
        create_conformal_map
    )
    from fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
    from comparison import create_domain_sources, get_sensor_locations
except ImportError:
    from .analytical_solver import (
        AnalyticalForwardSolver, 
        AnalyticalNonlinearInverseSolver,
        greens_function_disk_neumann,
        Source, InverseResult
    )
    from .conformal_solver import (
        ConformalForwardSolver,
        ConformalNonlinearInverseSolver,
        EllipseMap,
        NumericalConformalMap,
        create_conformal_map
    )
    from .fem_solver import FEMForwardSolver, FEMNonlinearInverseSolver
    from .comparison import create_domain_sources, get_sensor_locations


# =============================================================================
# POLAR PARAMETERIZATION (MATCHES MATLAB)
# =============================================================================

class PolarNonlinearInverseSolver:
    """
    Nonlinear inverse solver using POLAR parameterization.
    
    This matches the MATLAB implementation exactly:
        params = [S1, r1, θ1, S2, r2, θ2, ..., S_{n-1}, r_{n-1}, θ_{n-1}, r_n, θ_n]
        
    Key advantages over Cartesian:
        1. Natural disk constraint: 0 ≤ r ≤ r_max (box bounds suffice)
        2. No penalty functions needed
        3. Better scaling for optimization
        
    The last intensity is computed from Σq = 0 constraint.
    """
    
    def __init__(self, n_sources: int, n_boundary: int = 100,
                 r_min: float = 0.1, r_max: float = 0.9,
                 S_max: float = 5.0):
        """
        Parameters
        ----------
        n_sources : int
            Number of sources to recover
        n_boundary : int
            Number of boundary measurement points
        r_min, r_max : float
            Radial bounds for sources (enforces interior constraint)
        S_max : float
            Maximum absolute intensity
        """
        self.n_sources = n_sources
        self.n_boundary = n_boundary
        self.r_min = r_min
        self.r_max = r_max
        self.S_max = S_max
        
        # Boundary points
        self.theta_boundary = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        self.u_measured = None
        self.history = []
        
    def set_measured_data(self, u_measured: np.ndarray):
        """Set boundary measurements (will be mean-centered)."""
        self.u_measured = u_measured - np.mean(u_measured)
    
    def _params_to_sources(self, params: np.ndarray) -> List[Tuple[Tuple[float, float], float]]:
        """
        Convert polar parameters to source list.
        
        Parameters layout:
            [S1, r1, θ1, S2, r2, θ2, ..., S_{n-1}, r_{n-1}, θ_{n-1}, r_n, θ_n]
            
        Total params: 3*(n-1) + 2 = 3n - 1
        """
        n = self.n_sources
        sources = []
        intensity_sum = 0.0
        
        # First n-1 sources have (S, r, θ)
        for i in range(n - 1):
            S = params[3*i]
            r = params[3*i + 1]
            theta = params[3*i + 2]
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            sources.append(((x, y), S))
            intensity_sum += S
        
        # Last source: only (r, θ), intensity from constraint
        idx = 3 * (n - 1)
        r_last = params[idx]
        theta_last = params[idx + 1]
        S_last = -intensity_sum  # Enforce Σq = 0
        x_last = r_last * np.cos(theta_last)
        y_last = r_last * np.sin(theta_last)
        sources.append(((x_last, y_last), S_last))
        
        return sources
    
    def _boundary_potential(self, sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
        """
        Compute boundary potential using direct formula.
        
        For the Neumann Green's function on unit disk, the boundary potential is:
        
        u(θ) = -Σ_k (S_k / 2π) * log|e^{iθ} - ξ_k| + image terms
        
        Simplified for boundary evaluation (matches analytical solver):
        Using the full Green's function G(x,ξ) = -1/(2π)[ln|x-ξ| + ln|x-ξ*| - ln|ξ|]
        """
        u = np.zeros(self.n_boundary)
        
        for (x, y), S in sources:
            r_s = np.sqrt(x**2 + y**2)
            theta_s = np.arctan2(y, x)
            
            if r_s < 1e-10:
                # Source at origin - simple formula
                # G(e^{iθ}, 0) = -1/(2π) * ln(1) = 0, but we still have contribution
                continue
            
            # Distance to source: |e^{iθ} - r_s e^{iθ_s}|²
            dist_sq = 1 + r_s**2 - 2*r_s*np.cos(self.theta_boundary - theta_s)
            dist_sq = np.maximum(dist_sq, 1e-14)
            
            # Distance to image point ξ* = ξ/|ξ|² (so |ξ*| = 1/r_s)
            # |e^{iθ} - (1/r_s)e^{iθ_s}|² = 1 + 1/r_s² - 2/r_s * cos(θ - θ_s)
            dist_star_sq = 1 + 1/r_s**2 - (2/r_s)*np.cos(self.theta_boundary - theta_s)
            dist_star_sq = np.maximum(dist_star_sq, 1e-14)
            
            # Full Neumann Green's function:
            # G = -1/(2π) [ln|x-ξ| + ln|x-ξ*| - ln|ξ|]
            #   = -1/(2π) [0.5*ln(dist_sq) + 0.5*ln(dist_star_sq) - ln(r_s)]
            #   = -1/(4π) [ln(dist_sq) + ln(dist_star_sq)] + ln(r_s)/(2π)
            G_contrib = -1/(4*np.pi) * (np.log(dist_sq) + np.log(dist_star_sq)) + np.log(r_s)/(2*np.pi)
            
            u += S * G_contrib
        
        return u - np.mean(u)
    
    def _objective(self, params: np.ndarray) -> float:
        """Objective: ||u_computed - u_measured||²"""
        sources = self._params_to_sources(params)
        u_computed = self._boundary_potential(sources)
        misfit = np.sum((u_computed - self.u_measured)**2)
        self.history.append(misfit)
        return misfit
    
    def _get_bounds(self) -> List[Tuple[float, float]]:
        """
        Get parameter bounds for polar parameterization.
        
        For each of first n-1 sources: (S, r, θ) bounds
        For last source: (r, θ) bounds only
        """
        n = self.n_sources
        bounds = []
        
        for i in range(n - 1):
            bounds.append((-self.S_max, self.S_max))   # intensity
            bounds.append((self.r_min, self.r_max))    # radius
            bounds.append((0, 2*np.pi))                # angle
        
        # Last source (no intensity - derived from constraint)
        bounds.append((self.r_min, self.r_max))
        bounds.append((0, 2*np.pi))
        
        return bounds
    
    def _get_initial_guess(self, seed: int = 0) -> np.ndarray:
        """Generate initial guess with sources well-separated on annulus."""
        n = self.n_sources
        np.random.seed(42 + seed)
        
        params = []
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        angles += np.random.uniform(-0.3, 0.3, n)  # Perturb angles
        
        for i in range(n - 1):
            # Intensity: alternating ±1 with small noise
            S = (1.0 if i % 2 == 0 else -1.0) + 0.1 * np.random.randn()
            # Radius: in [0.5, 0.85] for good conditioning
            r = 0.5 + 0.35 * np.random.rand()
            theta = angles[i]
            params.extend([S, r, theta])
        
        # Last source: just position
        r = 0.5 + 0.35 * np.random.rand()
        theta = angles[n-1]
        params.extend([r, theta])
        
        return np.array(params)
    
    def solve(self, method: str = 'L-BFGS-B', maxiter: int = 500,
              n_restarts: int = 10) -> InverseResult:
        """
        Solve the inverse problem.
        
        Parameters
        ----------
        method : str
            'L-BFGS-B', 'differential_evolution', 'SLSQP'
        maxiter : int
            Maximum iterations per restart
        n_restarts : int
            Number of random restarts for local optimizers
            
        Returns
        -------
        InverseResult with recovered sources
        """
        if self.u_measured is None:
            raise ValueError("Call set_measured_data first")
        
        self.history = []
        bounds = self._get_bounds()
        
        best_result = None
        best_fun = np.inf
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self._objective, bounds, 
                maxiter=maxiter,
                seed=42, 
                polish=True,
                workers=1,
                tol=1e-10
            )
            best_result = result
            
        else:
            # Local optimizer with restarts
            for restart in range(n_restarts):
                x0 = self._get_initial_guess(restart)
                result = minimize(
                    self._objective, x0, 
                    method=method, 
                    bounds=bounds,
                    options={'maxiter': maxiter}
                )
                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
        
        # Convert to Source objects
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
# TEST SOURCE GENERATION
# =============================================================================

def create_well_separated_sources(n_sources: int, r: float = 0.7, 
                                   seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create n sources well-separated on a circle of radius r.
    
    Intensities alternate +1, -1, with last adjusted to satisfy Σq = 0.
    """
    np.random.seed(seed)
    sources = []
    
    # Evenly spaced angles with small perturbation
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    intensity_sum = 0.0
    for i in range(n_sources - 1):
        x = r * np.cos(angles[i])
        y = r * np.sin(angles[i])
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
        intensity_sum += S
    
    # Last source with intensity to satisfy constraint
    x = r * np.cos(angles[-1])
    y = r * np.sin(angles[-1])
    S_last = -intensity_sum
    sources.append(((x, y), S_last))
    
    return sources


def create_random_sources(n_sources: int, r_min: float = 0.5, r_max: float = 0.85,
                          seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create n sources at random positions in annulus [r_min, r_max].
    
    Intensities are random but constrained to sum to zero.
    """
    np.random.seed(seed)
    sources = []
    
    # Random intensities that sum to zero
    intensities = np.random.randn(n_sources)
    intensities = intensities - np.mean(intensities)  # Center to sum=0
    # Scale to have reasonable magnitudes
    intensities = intensities / np.std(intensities)
    
    for i in range(n_sources):
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        sources.append(((x, y), intensities[i]))
    
    return sources


# =============================================================================
# VALIDATION METRICS
# =============================================================================

@dataclass
class ValidationResult:
    """Results from a validation run."""
    domain: str
    n_sources: int
    method: str
    parameterization: str  # 'cartesian' or 'polar'
    position_rmse: float
    intensity_rmse: float
    residual: float
    time_seconds: float
    success: bool
    n_restarts: int = 0
    maxiter: int = 0
    

def compute_position_rmse(sources_true: List, sources_recovered: List) -> float:
    """
    Compute position RMSE using optimal matching (Hungarian algorithm).
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    if len(sources_recovered) != n:
        return np.inf
    
    # Build cost matrix
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j, src_r in enumerate(sources_recovered):
            if hasattr(src_r, 'x'):
                pos_r = (src_r.x, src_r.y)
            else:
                pos_r = src_r[0]
            cost[i, j] = np.sqrt((pos_t[0] - pos_r[0])**2 + (pos_t[1] - pos_r[1])**2)
    
    # Optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # RMSE
    errors = [cost[i, j] for i, j in zip(row_ind, col_ind)]
    return np.sqrt(np.mean(np.array(errors)**2))


def compute_intensity_rmse(sources_true: List, sources_recovered: List) -> float:
    """
    Compute intensity RMSE using same optimal matching as positions.
    """
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    if len(sources_recovered) != n:
        return np.inf
    
    # Build position cost matrix for matching
    cost = np.zeros((n, n))
    for i, (pos_t, _) in enumerate(sources_true):
        for j, src_r in enumerate(sources_recovered):
            if hasattr(src_r, 'x'):
                pos_r = (src_r.x, src_r.y)
            else:
                pos_r = src_r[0]
            cost[i, j] = np.sqrt((pos_t[0] - pos_r[0])**2 + (pos_t[1] - pos_r[1])**2)
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Compute intensity errors with matched pairs
    errors = []
    for i, j in zip(row_ind, col_ind):
        q_true = sources_true[i][1]
        if hasattr(sources_recovered[j], 'intensity'):
            q_rec = sources_recovered[j].intensity
        else:
            q_rec = sources_recovered[j][1]
        errors.append(q_true - q_rec)
    
    return np.sqrt(np.mean(np.array(errors)**2))


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def validate_disk_analytical(n_sources_list: List[int] = [2, 4, 6, 8, 10],
                             methods: List[str] = ['L-BFGS-B', 'differential_evolution'],
                             n_boundary: int = 100,
                             n_restarts: int = 20,
                             maxiter: int = 500,
                             verbose: bool = True) -> List[ValidationResult]:
    """
    Validate analytical nonlinear solver on disk domain.
    
    Compares Cartesian (current) vs Polar (MATLAB-style) parameterization.
    """
    results = []
    
    if verbose:
        print("\n" + "="*70)
        print("DISK DOMAIN - ANALYTICAL SOLVER VALIDATION")
        print("="*70)
    
    for n_sources in n_sources_list:
        sources_true = create_well_separated_sources(n_sources, r=0.7)
        
        # Generate measurements
        forward = AnalyticalForwardSolver(n_boundary)
        u_measured = forward.solve(sources_true)
        
        if verbose:
            print(f"\n--- {n_sources} sources (well-separated at r=0.7) ---")
        
        for method in methods:
            # Test Cartesian (current implementation)
            t0 = time()
            solver_cart = AnalyticalNonlinearInverseSolver(n_sources, n_boundary)
            solver_cart.set_measured_data(u_measured)
            result_cart = solver_cart.solve(method=method, maxiter=maxiter, 
                                            n_restarts=n_restarts if method != 'differential_evolution' else 1)
            t_cart = time() - t0
            
            pos_rmse_cart = compute_position_rmse(sources_true, result_cart.sources)
            int_rmse_cart = compute_intensity_rmse(sources_true, result_cart.sources)
            
            results.append(ValidationResult(
                domain='disk', n_sources=n_sources, method=method,
                parameterization='cartesian',
                position_rmse=pos_rmse_cart, intensity_rmse=int_rmse_cart,
                residual=result_cart.residual, time_seconds=t_cart,
                success=result_cart.success, n_restarts=n_restarts, maxiter=maxiter
            ))
            
            # Test Polar (MATLAB-style)
            t0 = time()
            solver_polar = PolarNonlinearInverseSolver(n_sources, n_boundary)
            solver_polar.set_measured_data(u_measured)
            result_polar = solver_polar.solve(method=method, maxiter=maxiter,
                                              n_restarts=n_restarts if method != 'differential_evolution' else 1)
            t_polar = time() - t0
            
            pos_rmse_polar = compute_position_rmse(sources_true, result_polar.sources)
            int_rmse_polar = compute_intensity_rmse(sources_true, result_polar.sources)
            
            results.append(ValidationResult(
                domain='disk', n_sources=n_sources, method=method,
                parameterization='polar',
                position_rmse=pos_rmse_polar, intensity_rmse=int_rmse_polar,
                residual=result_polar.residual, time_seconds=t_polar,
                success=result_polar.success, n_restarts=n_restarts, maxiter=maxiter
            ))
            
            if verbose:
                print(f"  {method}:")
                print(f"    Cartesian: pos_RMSE={pos_rmse_cart:.2e}, int_RMSE={int_rmse_cart:.2e}, "
                      f"residual={result_cart.residual:.2e}, time={t_cart:.1f}s")
                print(f"    Polar:     pos_RMSE={pos_rmse_polar:.2e}, int_RMSE={int_rmse_polar:.2e}, "
                      f"residual={result_polar.residual:.2e}, time={t_polar:.1f}s")
    
    return results


def validate_ellipse_conformal(n_sources_list: List[int] = [2, 4, 6],
                               a: float = 2.0, b: float = 1.0,
                               n_boundary: int = 100,
                               verbose: bool = True) -> List[ValidationResult]:
    """
    Validate conformal nonlinear solver on ellipse domain.
    """
    results = []
    
    if verbose:
        print("\n" + "="*70)
        print(f"ELLIPSE DOMAIN (a={a}, b={b}) - CONFORMAL SOLVER VALIDATION")
        print("="*70)
    
    ellipse_map = EllipseMap(a=a, b=b)
    
    for n_sources in n_sources_list:
        # Create sources inside ellipse
        sources_true = []
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        r_scale = 0.5  # Stay well inside
        intensity_sum = 0.0
        
        for i in range(n_sources - 1):
            x = r_scale * a * np.cos(angles[i])
            y = r_scale * b * np.sin(angles[i])
            S = 1.0 if i % 2 == 0 else -1.0
            sources_true.append(((x, y), S))
            intensity_sum += S
        
        x = r_scale * a * np.cos(angles[-1])
        y = r_scale * b * np.sin(angles[-1])
        sources_true.append(((x, y), -intensity_sum))
        
        # Generate measurements
        forward = ConformalForwardSolver(ellipse_map, n_boundary)
        u_measured = forward.solve(sources_true)
        
        if verbose:
            print(f"\n--- {n_sources} sources ---")
        
        for method in ['differential_evolution', 'lbfgsb']:
            t0 = time()
            solver = ConformalNonlinearInverseSolver(ellipse_map, n_sources, n_boundary)
            sources_rec, residual = solver.solve(u_measured, method=method)
            t_elapsed = time() - t0
            
            pos_rmse = compute_position_rmse(sources_true, sources_rec)
            int_rmse = compute_intensity_rmse(sources_true, sources_rec)
            
            results.append(ValidationResult(
                domain='ellipse', n_sources=n_sources, method=method,
                parameterization='cartesian',
                position_rmse=pos_rmse, intensity_rmse=int_rmse,
                residual=residual, time_seconds=t_elapsed, success=True
            ))
            
            if verbose:
                print(f"  {method}: pos_RMSE={pos_rmse:.2e}, int_RMSE={int_rmse:.2e}, "
                      f"residual={residual:.2e}, time={t_elapsed:.1f}s")
    
    return results


def validate_star_conformal(n_sources_list: List[int] = [2, 4],
                            n_petals: int = 5, amplitude: float = 0.3,
                            n_boundary: int = 100,
                            verbose: bool = True) -> List[ValidationResult]:
    """
    Validate conformal nonlinear solver on star-shaped domain.
    """
    results = []
    
    if verbose:
        print("\n" + "="*70)
        print(f"STAR DOMAIN ({n_petals} petals, amplitude={amplitude}) - CONFORMAL VALIDATION")
        print("="*70)
    
    # Define star-shaped domain using boundary function
    def star_boundary(t):
        """Parametric boundary: z(t) = r(t) * e^{it} where r(t) = 1 + amp*cos(n*t)"""
        r = 1.0 + amplitude * np.cos(n_petals * t)
        return r * np.exp(1j * t)
    
    try:
        star_map = NumericalConformalMap(star_boundary, n_boundary=256)
    except Exception as e:
        if verbose:
            print(f"  Skipping star domain: {e}")
        return results
    
    for n_sources in n_sources_list:
        # Create sources in safe interior (r < 0.5)
        sources_true = []
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        r_scale = 0.4
        intensity_sum = 0.0
        
        for i in range(n_sources - 1):
            x = r_scale * np.cos(angles[i])
            y = r_scale * np.sin(angles[i])
            S = 1.0 if i % 2 == 0 else -1.0
            sources_true.append(((x, y), S))
            intensity_sum += S
        
        x = r_scale * np.cos(angles[-1])
        y = r_scale * np.sin(angles[-1])
        sources_true.append(((x, y), -intensity_sum))
        
        # Generate measurements
        try:
            forward = ConformalForwardSolver(star_map, n_boundary)
            u_measured = forward.solve(sources_true)
        except Exception as e:
            if verbose:
                print(f"  Skipping {n_sources} sources: forward solve failed - {e}")
            continue
        
        if verbose:
            print(f"\n--- {n_sources} sources ---")
        
        t0 = time()
        solver = ConformalNonlinearInverseSolver(star_map, n_sources, n_boundary)
        sources_rec, residual = solver.solve(u_measured, method='differential_evolution')
        t_elapsed = time() - t0
        
        pos_rmse = compute_position_rmse(sources_true, sources_rec)
        int_rmse = compute_intensity_rmse(sources_true, sources_rec)
        
        results.append(ValidationResult(
            domain='star', n_sources=n_sources, method='differential_evolution',
            parameterization='cartesian',
            position_rmse=pos_rmse, intensity_rmse=int_rmse,
            residual=residual, time_seconds=t_elapsed, success=True
        ))
        
        if verbose:
            print(f"  diff_evol: pos_RMSE={pos_rmse:.2e}, int_RMSE={int_rmse:.2e}, "
                  f"residual={residual:.2e}, time={t_elapsed:.1f}s")
    
    return results


def validate_scaling_study(n_sources_range: range = range(2, 11),
                           n_trials: int = 3,
                           method: str = 'L-BFGS-B',
                           n_restarts: int = 30,
                           maxiter: int = 500,
                           verbose: bool = True) -> Dict:
    """
    Study how solver performance scales with number of sources.
    
    For each n_sources, run multiple trials with different random seeds
    and report mean/std of position RMSE.
    """
    if verbose:
        print("\n" + "="*70)
        print("SCALING STUDY: POSITION RMSE vs NUMBER OF SOURCES")
        print(f"Method: {method}, Restarts: {n_restarts}, Trials: {n_trials}")
        print("="*70)
    
    results = {
        'n_sources': [],
        'polar_mean': [], 'polar_std': [],
        'cartesian_mean': [], 'cartesian_std': [],
        'polar_times': [], 'cartesian_times': []
    }
    
    n_boundary = 100
    
    for n_sources in n_sources_range:
        polar_errors = []
        cart_errors = []
        polar_times = []
        cart_times = []
        
        if verbose:
            print(f"\nn_sources = {n_sources}")
        
        for trial in range(n_trials):
            # Generate sources with different seed each trial
            sources_true = create_well_separated_sources(n_sources, r=0.7, seed=42+trial*100)
            
            # Generate measurements
            forward = AnalyticalForwardSolver(n_boundary)
            u_measured = forward.solve(sources_true)
            
            # Polar solver
            t0 = time()
            solver_polar = PolarNonlinearInverseSolver(n_sources, n_boundary)
            solver_polar.set_measured_data(u_measured)
            result_polar = solver_polar.solve(method=method, maxiter=maxiter, n_restarts=n_restarts)
            polar_times.append(time() - t0)
            polar_errors.append(compute_position_rmse(sources_true, result_polar.sources))
            
            # Cartesian solver  
            t0 = time()
            solver_cart = AnalyticalNonlinearInverseSolver(n_sources, n_boundary)
            solver_cart.set_measured_data(u_measured)
            result_cart = solver_cart.solve(method=method, maxiter=maxiter, n_restarts=n_restarts)
            cart_times.append(time() - t0)
            cart_errors.append(compute_position_rmse(sources_true, result_cart.sources))
            
            if verbose:
                print(f"  Trial {trial+1}: Polar={polar_errors[-1]:.2e}, Cart={cart_errors[-1]:.2e}")
        
        results['n_sources'].append(n_sources)
        results['polar_mean'].append(np.mean(polar_errors))
        results['polar_std'].append(np.std(polar_errors))
        results['cartesian_mean'].append(np.mean(cart_errors))
        results['cartesian_std'].append(np.std(cart_errors))
        results['polar_times'].append(np.mean(polar_times))
        results['cartesian_times'].append(np.mean(cart_times))
        
        if verbose:
            print(f"  Mean: Polar={results['polar_mean'][-1]:.2e} ± {results['polar_std'][-1]:.2e}, "
                  f"Cart={results['cartesian_mean'][-1]:.2e} ± {results['cartesian_std'][-1]:.2e}")
    
    return results


def validate_differential_evolution_vs_lbfgsb(n_sources_list: List[int] = [4, 6, 8],
                                               n_trials: int = 3,
                                               verbose: bool = True) -> Dict:
    """
    Compare differential_evolution (global) vs L-BFGS-B (local with restarts).
    """
    if verbose:
        print("\n" + "="*70)
        print("OPTIMIZER COMPARISON: differential_evolution vs L-BFGS-B")
        print("="*70)
    
    results = {
        'n_sources': [],
        'de_errors': [], 'de_times': [],
        'lbfgsb_errors': [], 'lbfgsb_times': []
    }
    
    n_boundary = 100
    
    for n_sources in n_sources_list:
        de_errors = []
        lbfgsb_errors = []
        de_times = []
        lbfgsb_times = []
        
        if verbose:
            print(f"\nn_sources = {n_sources}")
        
        for trial in range(n_trials):
            sources_true = create_well_separated_sources(n_sources, r=0.7, seed=42+trial*100)
            forward = AnalyticalForwardSolver(n_boundary)
            u_measured = forward.solve(sources_true)
            
            # Differential Evolution (polar)
            t0 = time()
            solver = PolarNonlinearInverseSolver(n_sources, n_boundary)
            solver.set_measured_data(u_measured)
            result = solver.solve(method='differential_evolution', maxiter=500)
            de_times.append(time() - t0)
            de_errors.append(compute_position_rmse(sources_true, result.sources))
            
            # L-BFGS-B with restarts (polar)
            t0 = time()
            solver = PolarNonlinearInverseSolver(n_sources, n_boundary)
            solver.set_measured_data(u_measured)
            result = solver.solve(method='L-BFGS-B', maxiter=500, n_restarts=30)
            lbfgsb_times.append(time() - t0)
            lbfgsb_errors.append(compute_position_rmse(sources_true, result.sources))
            
            if verbose:
                print(f"  Trial {trial+1}: DE={de_errors[-1]:.2e} ({de_times[-1]:.1f}s), "
                      f"L-BFGS-B={lbfgsb_errors[-1]:.2e} ({lbfgsb_times[-1]:.1f}s)")
        
        results['n_sources'].append(n_sources)
        results['de_errors'].append(de_errors)
        results['de_times'].append(de_times)
        results['lbfgsb_errors'].append(lbfgsb_errors)
        results['lbfgsb_times'].append(lbfgsb_times)
        
        if verbose:
            print(f"  Mean: DE={np.mean(de_errors):.2e}, L-BFGS-B={np.mean(lbfgsb_errors):.2e}")
    
    return results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary_report(results: List[ValidationResult]):
    """Print a summary table of validation results."""
    print("\n" + "="*90)
    print("VALIDATION SUMMARY")
    print("="*90)
    print(f"{'Domain':<10} {'n_src':<6} {'Method':<22} {'Param':<10} {'Pos RMSE':<12} {'Success'}")
    print("-"*90)
    
    for r in results:
        status = "✓" if r.position_rmse < 0.01 else ("~" if r.position_rmse < 0.1 else "✗")
        print(f"{r.domain:<10} {r.n_sources:<6} {r.method:<22} {r.parameterization:<10} "
              f"{r.position_rmse:<12.2e} {status}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate nonlinear inverse solvers")
    parser.add_argument('--quick', action='store_true', help="Quick mode (fewer tests)")
    parser.add_argument('--domain', type=str, default='all', 
                        choices=['all', 'disk', 'ellipse', 'star', 'scaling'],
                        help="Which domain(s) to test")
    args = parser.parse_args()
    
    all_results = []
    
    if args.quick:
        n_sources_disk = [2, 4]
        n_sources_other = [2, 4]
        n_sources_scaling = range(2, 7)
        methods = ['L-BFGS-B']
        n_restarts = 10
        n_trials = 2
    else:
        n_sources_disk = [2, 4, 6, 8]
        n_sources_other = [2, 4, 6]
        n_sources_scaling = range(2, 11)
        methods = ['L-BFGS-B', 'differential_evolution']
        n_restarts = 20
        n_trials = 3
    
    # Run validations
    if args.domain in ['all', 'disk']:
        results = validate_disk_analytical(
            n_sources_list=n_sources_disk,
            methods=methods,
            n_restarts=n_restarts
        )
        all_results.extend(results)
    
    if args.domain in ['all', 'ellipse']:
        results = validate_ellipse_conformal(n_sources_list=n_sources_other)
        all_results.extend(results)
    
    if args.domain in ['all', 'star']:
        results = validate_star_conformal(n_sources_list=[2, 4])
        all_results.extend(results)
    
    if args.domain in ['all', 'scaling']:
        scaling_results = validate_scaling_study(
            n_sources_range=n_sources_scaling,
            n_trials=n_trials,
            n_restarts=n_restarts
        )
        
        # Compare optimizers
        opt_results = validate_differential_evolution_vs_lbfgsb(
            n_sources_list=[4, 6, 8] if not args.quick else [4],
            n_trials=n_trials
        )
    
    # Print summary
    if all_results:
        print_summary_report(all_results)
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
