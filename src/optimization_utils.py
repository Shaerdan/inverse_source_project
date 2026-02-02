"""
Optimization Utilities for Inverse Source Localization
=======================================================

This module contains key fixes that enable MATLAB-equivalent convergence:

1. push_to_interior() - Pushes initial points away from bounds to avoid gradient blow-up
2. generate_diverse_starts() - Multiple initialization strategies to avoid bad local minima
3. Direct boundary formulas for disk domains (polar coordinates)

These fixes address the root cause of Python optimizers failing where MATLAB succeeds:
MATLAB's fmincon automatically pushes initial points to the interior, while Python
optimizers do not.

Key Insight from Debug Summary:
- Linear solvers (L1/L2/TV on grid) are fundamentally ill-posed (mutual coherence ~0.99)
- Nonlinear continuous approach works for well-separated sources (RMSE < 1e-5)
- Polar parameterization naturally constrains to disk via bounds on r

Author: Based on debugging session matching MATLAB fmincon behavior
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from typing import List, Tuple, Callable, Optional


def push_to_interior(x0: np.ndarray, bounds: List[Tuple[float, float]], 
                     margin: float = 0.1) -> np.ndarray:
    """
    Push initial point away from bounds to avoid gradient blow-up.
    
    MATLAB's fmincon automatically does this:
    "fmincon resets x0 components that are on or outside bounds lb or ub 
    to values strictly between the bounds"
    
    Without this fix, linspace-style initializations put many values exactly
    at bounds, causing gradient norms of ~10^16 and immediate termination.
    
    Parameters
    ----------
    x0 : array
        Initial parameter values
    bounds : list of (lower, upper)
        Bounds for each parameter
    margin : float
        Fraction of bound range to use as margin (default 0.1 = 10%)
        
    Returns
    -------
    x : array
        Initial point pushed to interior
    """
    x = np.asarray(x0).copy()
    for i, (lb, ub) in enumerate(bounds):
        if lb is None or ub is None:
            continue  # Skip unbounded parameters
        
        range_i = ub - lb
        interior_lb = lb + margin * range_i
        interior_ub = ub - margin * range_i
        
        x[i] = np.clip(x[i], interior_lb, interior_ub)
    
    return x


def generate_spread_init(n_sources: int, domain_type: str = 'disk',
                         domain_params: dict = None, seed: int = 42) -> np.ndarray:
    """
    Generate sources spread evenly in a pattern.
    
    This initialization strategy helps avoid local minima by starting
    with well-separated source positions.
    
    Parameters
    ----------
    n_sources : int
        Number of sources
    domain_type : str
        'disk', 'ellipse', 'polygon', 'square', 'rectangle'
    domain_params : dict
        Domain-specific parameters
    seed : int
        Random seed
        
    Returns
    -------
    x0 : array
        Initial parameters [x0, y0, x1, y1, ..., q0, q1, ...]
    """
    np.random.seed(seed)
    n_params = 3 * n_sources
    x0 = np.zeros(n_params)
    
    # Get domain center and scale
    if domain_type == 'disk':
        center_x, center_y = 0.0, 0.0
        scale = 0.6  # Use r=0.6 for well-separated sources
    elif domain_type == 'ellipse' and domain_params:
        center_x, center_y = 0.0, 0.0
        a = domain_params.get('a', 1.0)
        b = domain_params.get('b', 1.0)
        scale = 0.6 * min(a, b)
    elif domain_type in ['polygon', 'square', 'rectangle'] and domain_params:
        vertices = domain_params.get('vertices', [])
        if vertices:
            vertices = np.array(vertices)
            center_x = np.mean(vertices[:, 0])
            center_y = np.mean(vertices[:, 1])
            scale = 0.4 * min(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
        else:
            center_x, center_y, scale = 0.0, 0.0, 0.5
    else:
        center_x, center_y, scale = 0.0, 0.0, 0.5
    
    # Spread sources evenly around center
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += seed * 0.1  # Small offset for different seeds
    
    for i in range(n_sources):
        if domain_type == 'ellipse' and domain_params:
            a = domain_params.get('a', 1.0)
            b = domain_params.get('b', 1.0)
            x0[2*i] = center_x + 0.6 * a * np.cos(angles[i])
            x0[2*i + 1] = center_y + 0.6 * b * np.sin(angles[i])
        else:
            x0[2*i] = center_x + scale * np.cos(angles[i])
            x0[2*i + 1] = center_y + scale * np.sin(angles[i])
    
    # Alternating intensities (satisfies sum=0 approximately)
    for i in range(n_sources):
        x0[2*n_sources + i] = 0.5 * (1 if i % 2 == 0 else -1)
    
    # Center intensities for exact zero-sum
    intensities = x0[2*n_sources:3*n_sources]
    intensities -= np.mean(intensities)
    x0[2*n_sources:3*n_sources] = intensities
    
    return x0


def generate_random_init(n_sources: int, bounds: List[Tuple[float, float]],
                         seed: int = 42) -> np.ndarray:
    """
    Generate random initialization within bounds.
    
    Uses inner 60% of bounds to avoid edge effects.
    """
    np.random.seed(seed)
    n_params = 3 * n_sources
    x0 = np.zeros(n_params)
    
    # Random positions within inner region of bounds
    for i in range(n_sources):
        x_lb, x_ub = bounds[2*i]
        y_lb, y_ub = bounds[2*i + 1]
        
        # Use inner 60% of bounds
        x_range = x_ub - x_lb
        y_range = y_ub - y_lb
        x0[2*i] = x_lb + 0.2*x_range + 0.6*x_range * np.random.rand()
        x0[2*i + 1] = y_lb + 0.2*y_range + 0.6*y_range * np.random.rand()
    
    # Random intensities
    for i in range(n_sources):
        x0[2*n_sources + i] = np.random.randn()
    
    # Center intensities for zero-sum
    intensities = x0[2*n_sources:3*n_sources]
    intensities -= np.mean(intensities)
    x0[2*n_sources:3*n_sources] = intensities
    
    return x0


# =============================================================================
# DIRECT BOUNDARY FORMULAS (POLAR COORDINATES FOR DISK)
# =============================================================================

def boundary_potential_disk_polar(theta_boundary: np.ndarray, 
                                   sources_polar: List[Tuple[float, float, float]]) -> np.ndarray:
    """
    Compute boundary potential using direct formula (polar coordinates).
    
    This is the EXACT boundary solution for point sources in the unit disk
    with Neumann boundary conditions (MATLAB-equivalent formula):
    
        u(θ) = Σₖ (Sₖ / 2π) * log(1 + rₖ² - 2*rₖ*cos(θ - θₖ))
    
    Parameters
    ----------
    theta_boundary : array, shape (n,)
        Angles of boundary measurement points
    sources_polar : list of (S, r, theta)
        Source intensities and polar positions
        
    Returns
    -------
    u : array, shape (n,)
        Boundary potential values (mean-centered)
    """
    u = np.zeros_like(theta_boundary)
    
    for S, r_s, theta_s in sources_polar:
        # Distance squared from boundary point e^{iθ} to source r_s*e^{iθ_s}:
        # |e^{iθ} - r_s*e^{iθ_s}|² = 1 + r_s² - 2*r_s*cos(θ - θ_s)
        dist_sq = 1 + r_s**2 - 2*r_s*np.cos(theta_boundary - theta_s)
        dist_sq = np.maximum(dist_sq, 1e-14)  # Avoid log(0)
        
        u += (S / (2*np.pi)) * np.log(dist_sq)
    
    return u - np.mean(u)  # Mean-center


def boundary_potential_disk_cartesian(theta_boundary: np.ndarray,
                                       sources: List[Tuple[Tuple[float, float], float]]) -> np.ndarray:
    """
    Compute boundary potential from Cartesian source positions.
    
    Same formula as polar but accepts ((x, y), S) format.
    """
    u = np.zeros_like(theta_boundary)
    
    for (x, y), S in sources:
        r_s = np.sqrt(x**2 + y**2)
        theta_s = np.arctan2(y, x)
        
        dist_sq = 1 + r_s**2 - 2*r_s*np.cos(theta_boundary - theta_s)
        dist_sq = np.maximum(dist_sq, 1e-14)
        
        u += (S / (2*np.pi)) * np.log(dist_sq)
    
    return u - np.mean(u)


# =============================================================================
# MULTISTART SOLVERS
# =============================================================================

def solve_slsqp_multistart(
    objective: Callable,
    bounds: List[Tuple[float, float]],
    n_sources: int,
    constraint_fun: Callable = None,
    n_starts: int = 5,
    maxiter: int = 10000,
    ftol: float = 1e-14,
    domain_type: str = 'disk',
    domain_params: dict = None,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[np.ndarray, float, dict]:
    """
    Solve nonlinear optimization with SLSQP using multistart.
    
    SLSQP (Sequential Least Squares Programming) is recommended because:
    1. It handles equality constraints well (intensity sum = 0)
    2. Works better than L-BFGS-B with barriers for this problem
    3. Closest to MATLAB's fmincon SQP component
    
    Parameters
    ----------
    objective : callable
        Objective function f(x) -> float
    bounds : list of (lower, upper)
        Bounds for each parameter
    n_sources : int
        Number of sources
    constraint_fun : callable, optional
        Constraint function g(x) = 0 (equality constraint)
    n_starts : int
        Number of random restarts
    maxiter : int
        Maximum iterations per start
    ftol : float
        Function tolerance for convergence
    domain_type : str
        Domain type for initialization
    domain_params : dict
        Domain parameters for initialization
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    x_best : array
        Best solution found
    f_best : float
        Best objective value
    info : dict
        Additional information
    """
    best_x = None
    best_f = np.inf
    best_idx = -1
    
    # Set up constraint
    constraints = []
    if constraint_fun is not None:
        constraints.append({
            'type': 'eq',
            'fun': constraint_fun,
        })
    
    # Generate diverse starting points
    starts = []
    
    # Strategy 1: Spread initialization
    x0_spread = generate_spread_init(n_sources, domain_type, domain_params, seed)
    starts.append(push_to_interior(x0_spread, bounds))
    
    # Strategy 2+: Random initializations
    for i in range(max(0, n_starts - 1)):
        x0_random = generate_random_init(n_sources, bounds, seed + i * 100)
        starts.append(push_to_interior(x0_random, bounds))
    
    for idx, x0 in enumerate(starts):
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
            )
            
            if verbose:
                print(f"  Start {idx+1}: obj = {result.fun:.2e}")
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                best_idx = idx
                
        except Exception as e:
            if verbose:
                print(f"  Start {idx+1}: FAILED - {e}")
    
    info = {
        'n_starts_tried': len(starts),
        'best_start_idx': best_idx,
        'success': best_x is not None,
    }
    
    return best_x, best_f, info


# =============================================================================
# POLAR COORDINATE SOLVER FOR DISK (MATLAB-EQUIVALENT)
# =============================================================================

def solve_disk_polar(
    theta_boundary: np.ndarray,
    u_measured: np.ndarray,
    n_sources: int,
    n_starts: int = 5,
    maxiter: int = 10000,
    r_min: float = 0.05,
    r_max: float = 0.95,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[List[Tuple[Tuple[float, float], float]], float]:
    """
    Solve inverse source problem on disk using polar parameterization.
    
    This is the most robust method for disk domains, matching MATLAB performance.
    
    Uses polar coordinates (S, r, θ) which naturally constrains sources to disk
    via bounds on r, avoiding the need for barrier functions or nonlinear constraints.
    
    From debug summary:
        "Source parameters: [S1, r1, θ1, S2, r2, θ2, ...]
         Forward model: phi = (S/(2*pi)) * log(1 + r² - 2*r*cos(θ - θ_s))"
    
    Parameters
    ----------
    theta_boundary : array
        Angles of boundary measurement points
    u_measured : array
        Measured boundary potential
    n_sources : int
        Number of sources to recover
    n_starts : int
        Number of multistart attempts
    maxiter : int
        Maximum iterations
    r_min : float
        Minimum source radius (default 0.05)
    r_max : float
        Maximum source radius (default 0.95)
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    sources : list of ((x, y), intensity)
        Recovered sources in Cartesian coordinates
    residual : float
        Final residual (sqrt of sum of squares)
    """
    # Center measurements
    u_meas = u_measured - np.mean(u_measured)
    
    # Parameter layout: [S0, r0, θ0, S1, r1, θ1, ...]
    n_params = 3 * n_sources
    
    # Bounds: S in [-5, 5], r in [r_min, r_max], θ in [-π, π]
    bounds = []
    for i in range(n_sources):
        bounds.append((-5.0, 5.0))    # S
        bounds.append((r_min, r_max))  # r (positive, inside disk)
        bounds.append((-np.pi, np.pi))  # θ
    
    def forward_polar(params):
        """Compute boundary potential from polar parameters."""
        u = np.zeros_like(theta_boundary)
        for i in range(n_sources):
            S = params[3*i]
            r = params[3*i + 1]
            th = params[3*i + 2]
            
            arg = 1 + r**2 - 2*r*np.cos(theta_boundary - th)
            arg = np.maximum(arg, 1e-30)
            u += (S / (2*np.pi)) * np.log(arg)
        
        return u - np.mean(u)
    
    def objective(params):
        u_computed = forward_polar(params)
        return np.sum((u_computed - u_meas)**2)
    
    def intensity_constraint(params):
        """Sum of intensities should equal zero."""
        return sum(params[3*i] for i in range(n_sources))
    
    # Generate diverse starts for polar coordinates
    starts = []
    np.random.seed(seed)
    
    # Start 1: Evenly spread at r=0.6
    x0_spread = np.zeros(n_params)
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    for i in range(n_sources):
        x0_spread[3*i] = 0.5 * (1 if i % 2 == 0 else -1)  # Alternating S
        x0_spread[3*i + 1] = 0.6  # r
        x0_spread[3*i + 2] = angles[i]  # θ
    # Adjust last intensity for sum=0
    x0_spread[3*(n_sources-1)] = -sum(x0_spread[3*i] for i in range(n_sources-1))
    starts.append(push_to_interior(x0_spread, bounds))
    
    # Start 2+: Random
    for j in range(n_starts - 1):
        np.random.seed(seed + j * 100)
        x0_rand = np.zeros(n_params)
        for i in range(n_sources):
            x0_rand[3*i] = np.random.uniform(-2, 2)
            x0_rand[3*i + 1] = np.random.uniform(0.3, 0.8)
            x0_rand[3*i + 2] = np.random.uniform(-np.pi, np.pi)
        # Adjust last intensity for sum=0
        x0_rand[3*(n_sources-1)] = -sum(x0_rand[3*i] for i in range(n_sources-1))
        starts.append(push_to_interior(x0_rand, bounds))
    
    # Solve with SLSQP multistart
    best_x = None
    best_f = np.inf
    
    constraints = [{'type': 'eq', 'fun': intensity_constraint}]
    
    for idx, x0 in enumerate(starts):
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
            )
            
            if verbose:
                print(f"  Start {idx+1}: obj = {result.fun:.2e}")
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception as e:
            if verbose:
                print(f"  Start {idx+1}: FAILED - {e}")
    
    # Convert polar to Cartesian sources
    sources = []
    if best_x is not None:
        # Center intensities
        intensities = np.array([best_x[3*i] for i in range(n_sources)])
        intensities = intensities - np.mean(intensities)
        
        for i in range(n_sources):
            r = best_x[3*i + 1]
            th = best_x[3*i + 2]
            x = r * np.cos(th)
            y = r * np.sin(th)
            sources.append(((x, y), intensities[i]))
    
    return sources, np.sqrt(best_f) if best_f < np.inf else np.inf


# =============================================================================
# WELL-SEPARATED SOURCE GENERATION (FOR TESTING)
# =============================================================================

def create_well_separated_sources(n_sources: int, r_range: Tuple[float, float] = (0.6, 0.9),
                                   seed: int = 42) -> List[Tuple[Tuple[float, float], float]]:
    """
    Create sources with guaranteed minimum separation.
    
    From debug summary:
        "Sources with intensities: alternating +1, -1
         Angles: evenly spaced with small perturbations
         Radii: r in [0.6, 0.9]"
    
    These are "easy" test cases where nonlinear solvers should achieve ~0 error.
    
    Parameters
    ----------
    n_sources : int
        Number of sources to create
    r_range : tuple
        (r_min, r_max) for source radii
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    sources : list of ((x, y), intensity)
        Source positions and intensities
    """
    np.random.seed(seed)
    
    sources = []
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, n_sources)  # Small perturbation
    
    for i, theta in enumerate(angles):
        r = np.random.uniform(r_range[0], r_range[1])
        x, y = r * np.cos(theta), r * np.sin(theta)
        S = 1.0 if i % 2 == 0 else -1.0
        sources.append(((x, y), S))
    
    # Adjust last intensity for sum = 0
    total = sum(s[1] for s in sources)
    sources[-1] = (sources[-1][0], sources[-1][1] - total)
    
    return sources


def compute_source_metrics(sources_true: List, sources_rec: List) -> dict:
    """
    Compute metrics comparing recovered vs true sources.
    
    Uses Hungarian algorithm for optimal matching.
    
    Returns
    -------
    dict with keys:
        'position_rmse': Position RMSE after optimal matching
        'intensity_rmse': Intensity RMSE after optimal matching
        'matched_pairs': List of (true_idx, rec_idx) pairs
    """
    from scipy.optimize import linear_sum_assignment
    
    n_true = len(sources_true)
    n_rec = len(sources_rec)
    
    if n_rec == 0:
        return {'position_rmse': np.inf, 'intensity_rmse': np.inf, 'matched_pairs': []}
    
    # Build cost matrix (position distances)
    cost_matrix = np.zeros((n_true, n_rec))
    for i, (pos_t, _) in enumerate(sources_true):
        for j, (pos_r, _) in enumerate(sources_rec):
            dx = pos_t[0] - pos_r[0]
            dy = pos_t[1] - pos_r[1]
            cost_matrix[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Compute metrics
    pos_errors = []
    int_errors = []
    matched_pairs = []
    
    for i, j in zip(row_ind, col_ind):
        pos_t, q_t = sources_true[i]
        pos_r, q_r = sources_rec[j]
        
        dx = pos_t[0] - pos_r[0]
        dy = pos_t[1] - pos_r[1]
        pos_errors.append(dx**2 + dy**2)
        int_errors.append((q_t - q_r)**2)
        matched_pairs.append((i, j))
    
    return {
        'position_rmse': np.sqrt(np.mean(pos_errors)),
        'intensity_rmse': np.sqrt(np.mean(int_errors)),
        'matched_pairs': matched_pairs
    }
