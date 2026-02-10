"""
Depth-Weighted Linear Solvers for Inverse Source Problems

This module provides weighted versions of L2, L1, and TV regularization
that correct for the boundary bias in Green's function based linear methods.

The key insight is that Green's function column norms vary ~20x between
boundary and center sources, causing standard regularization to push
intensity toward the boundary. Depth weighting w_j = (1 - rho_j)^(-beta)
compensates for this sensitivity variation.

Key functions:
- compute_conformal_radii_disk/general: Get conformal radius for each grid point
- compute_depth_weights: Compute w_j = (1 - rho_j)^(-beta)
- solve_l2_weighted: Weighted Tikhonov (L2) regularization
- solve_l1_weighted_admm: Weighted L1 via ADMM
- solve_tv_weighted: Weighted Total Variation
- compute_l_curve_weighted: Alpha selection with discrepancy principle option

Usage:
    radii = compute_conformal_radii_disk(grid_points)
    weights = compute_depth_weights(radii, beta=1.0)
    alpha, _, _ = compute_l_curve_weighted(G, u, 'l1', weights, noise_level=sigma)
    q = solve_l1_weighted_admm(G, u, alpha, A_eq, b_eq, weights)

Reference:
    The depth weighting corrects for the decay ||G_j|| ~ (1 - rho_j) observed
    in Green's function matrices. With beta=1.0, the weighted penalty
    w_j * |q_j| has approximately uniform sensitivity across depths.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.optimize import minimize_scalar
import warnings


# =============================================================================
# Conformal Radius Computation
# =============================================================================

def compute_conformal_radii_disk(grid_points):
    """
    Compute conformal radii for grid points in the unit disk.
    
    For the disk, conformal radius = Euclidean distance from origin.
    
    Parameters
    ----------
    grid_points : array-like of complex
        Grid point positions as complex numbers (x + iy)
    
    Returns
    -------
    radii : ndarray
        Conformal radius for each grid point, in [0, 1)
    """
    grid_points = np.asarray(grid_points)
    radii = np.abs(grid_points)
    return radii


def compute_conformal_radii_general(grid_points, conformal_map):
    """
    Compute conformal radii for grid points in a general domain.
    
    Parameters
    ----------
    grid_points : array-like of complex
        Grid point positions in physical coordinates
    conformal_map : object
        Conformal map object with to_disk(z) method
    
    Returns
    -------
    radii : ndarray
        Conformal radius for each grid point
    """
    grid_points = np.asarray(grid_points)
    radii = np.array([np.abs(conformal_map.to_disk(z)) for z in grid_points])
    return radii


# =============================================================================
# Depth Weighting
# =============================================================================

def compute_depth_weights(conformal_radii, beta=1.0, rho_max=0.98):
    """
    Compute depth-dependent weights for regularization.
    
    The weight w_j = (1 - rho_j)^(-beta) compensates for the fact that
    Green's function column norms decay roughly as (1 - rho_j) near the
    boundary. With beta=1.0, this approximately equalizes sensitivity.
    
    Parameters
    ----------
    conformal_radii : ndarray
        Conformal radius for each grid point, in [0, 1)
    beta : float
        Weighting exponent (default 1.0, empirically validated)
    rho_max : float
        Maximum radius for clamping to avoid division by zero
    
    Returns
    -------
    weights : ndarray
        Weight w_j for each grid point
    """
    rho = np.asarray(conformal_radii)
    rho_clamped = np.clip(rho, 0, rho_max)
    weights = (1.0 - rho_clamped) ** (-beta)
    return weights


def select_beta_heuristic(G, conformal_radii, n_bins=10):
    """
    Heuristically select beta by fitting column norm vs (1-rho) relationship.
    
    This fits log(||G_j||) ~ -beta * log(1 - rho_j) and returns the slope.
    
    Parameters
    ----------
    G : ndarray (M, Mg)
        Green's function matrix
    conformal_radii : ndarray (Mg,)
        Conformal radii of grid points
    n_bins : int
        Number of bins for fitting
    
    Returns
    -------
    beta : float
        Estimated weighting exponent
    """
    col_norms = np.linalg.norm(G, axis=0)
    
    # Bin by radius
    rho = np.asarray(conformal_radii)
    bins = np.linspace(0.1, 0.95, n_bins + 1)
    
    log_one_minus_rho = []
    log_norm = []
    
    for i in range(n_bins):
        mask = (rho >= bins[i]) & (rho < bins[i+1])
        if np.sum(mask) > 0:
            mean_rho = np.mean(rho[mask])
            mean_norm = np.mean(col_norms[mask])
            if mean_rho < 0.99 and mean_norm > 0:
                log_one_minus_rho.append(np.log(1 - mean_rho))
                log_norm.append(np.log(mean_norm))
    
    if len(log_one_minus_rho) < 3:
        return 1.0  # Default
    
    # Linear regression: log(norm) = a + beta * log(1-rho)
    x = np.array(log_one_minus_rho)
    y = np.array(log_norm)
    
    # Least squares
    A = np.vstack([x, np.ones_like(x)]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    beta = result[0][0]
    
    # Clamp to reasonable range
    beta = np.clip(beta, 0.5, 2.0)
    
    return beta


# =============================================================================
# Weighted Solvers
# =============================================================================

def solve_l2_weighted(G, u, alpha, A_eq, b_eq, weights=None):
    """
    Solve weighted Tikhonov (L2) regularization problem.
    
    Minimizes: ||Gq - u||^2 + alpha * sum_j w_j * q_j^2
    Subject to: A_eq @ q = b_eq
    
    Parameters
    ----------
    G : ndarray (M, Mg)
        Forward matrix
    u : ndarray (M,)
        Measurement data
    alpha : float
        Regularization parameter
    A_eq : ndarray (n_eq, Mg)
        Equality constraint matrix
    b_eq : ndarray (n_eq,)
        Equality constraint RHS
    weights : ndarray (Mg,) or None
        Depth weights (None = uniform)
    
    Returns
    -------
    q : ndarray (Mg,)
        Solution intensity vector
    """
    M, Mg = G.shape
    
    if weights is None:
        weights = np.ones(Mg)
    
    W = np.diag(weights)
    
    # KKT system for constrained weighted L2
    # [G'G + alpha*W,  A_eq'] [q]   = [G'u]
    # [A_eq,           0    ] [lam]   [b_eq]
    
    GtG = G.T @ G
    Gtu = G.T @ u
    
    n_eq = A_eq.shape[0]
    
    # Build KKT matrix
    KKT = np.zeros((Mg + n_eq, Mg + n_eq))
    KKT[:Mg, :Mg] = GtG + alpha * W
    KKT[:Mg, Mg:] = A_eq.T
    KKT[Mg:, :Mg] = A_eq
    
    rhs = np.zeros(Mg + n_eq)
    rhs[:Mg] = Gtu
    rhs[Mg:] = b_eq
    
    try:
        sol = np.linalg.solve(KKT, rhs)
        q = sol[:Mg]
    except np.linalg.LinAlgError:
        # Fallback: use lstsq
        sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
        q = sol[:Mg]
    
    return q


def solve_l1_weighted_admm(G, u, alpha, A_eq, b_eq, weights=None,
                            rho_admm=1.0, max_iter=1000, tol=1e-6):
    """
    Solve weighted L1 regularization via ADMM.
    
    Minimizes: ||Gq - u||^2 + alpha * sum_j w_j * |q_j|
    Subject to: A_eq @ q = b_eq
    
    Uses ADMM splitting: min ||Gq - u||^2 + alpha * ||W @ z||_1
                         s.t. q = z, A_eq @ q = b_eq
    
    Parameters
    ----------
    G : ndarray (M, Mg)
        Forward matrix
    u : ndarray (M,)
        Measurement data
    alpha : float
        Regularization parameter
    A_eq : ndarray (n_eq, Mg)
        Equality constraint matrix
    b_eq : ndarray (n_eq,)
        Equality constraint RHS
    weights : ndarray (Mg,) or None
        Depth weights
    rho_admm : float
        ADMM penalty parameter
    max_iter : int
        Maximum ADMM iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    q : ndarray (Mg,)
        Solution intensity vector
    """
    M, Mg = G.shape
    
    if weights is None:
        weights = np.ones(Mg)
    
    n_eq = A_eq.shape[0]
    
    # Initialize
    q = np.zeros(Mg)
    z = np.zeros(Mg)
    y = np.zeros(Mg)  # Dual for q = z
    lam = np.zeros(n_eq)  # Dual for A_eq @ q = b_eq
    
    # Precompute
    GtG = G.T @ G
    Gtu = G.T @ u
    
    # For q-update: solve (G'G + rho*I) q + A_eq' lam = G'u + rho*(z - y)
    # Combined with A_eq @ q = b_eq
    
    # Build static part of KKT
    KKT_static = np.zeros((Mg + n_eq, Mg + n_eq))
    KKT_static[:Mg, :Mg] = GtG + rho_admm * np.eye(Mg)
    KKT_static[:Mg, Mg:] = A_eq.T
    KKT_static[Mg:, :Mg] = A_eq
    
    # Factorize for speed
    try:
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(KKT_static)
        use_lu = True
    except:
        use_lu = False
    
    for it in range(max_iter):
        q_old = q.copy()
        
        # q-update
        rhs = np.zeros(Mg + n_eq)
        rhs[:Mg] = Gtu + rho_admm * (z - y)
        rhs[Mg:] = b_eq
        
        if use_lu:
            sol = lu_solve((lu, piv), rhs)
        else:
            sol = np.linalg.solve(KKT_static, rhs)
        
        q = sol[:Mg]
        lam = sol[Mg:]
        
        # z-update: soft thresholding with weighted threshold
        # z = argmin alpha * w_j |z_j| + (rho/2) ||z - (q + y)||^2
        # Solution: soft_threshold(q + y, alpha * w / rho)
        v = q + y
        threshold = alpha * weights / rho_admm
        z = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        
        # y-update
        y = y + q - z
        
        # Check convergence
        primal_res = np.linalg.norm(q - z)
        dual_res = rho_admm * np.linalg.norm(q - q_old)
        
        if primal_res < tol and dual_res < tol:
            break
    
    return q


def solve_tv_weighted(G, u, alpha, A_eq, b_eq, weights=None,
                       rho_admm=1.0, max_iter=1000, tol=1e-6):
    """
    Solve weighted Total Variation regularization via ADMM.
    
    For a 1D ordered grid, TV = sum_j |q_{j+1} - q_j|.
    Weighted TV uses average weights at edges: w_edge = (w_j + w_{j+1})/2.
    
    Parameters
    ----------
    G : ndarray (M, Mg)
        Forward matrix
    u : ndarray (M,)
        Measurement data
    alpha : float
        Regularization parameter
    A_eq : ndarray (n_eq, Mg)
        Equality constraint matrix
    b_eq : ndarray (n_eq,)
        Equality constraint RHS
    weights : ndarray (Mg,) or None
        Depth weights for grid points
    rho_admm : float
        ADMM penalty parameter
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    q : ndarray (Mg,)
        Solution intensity vector
    """
    M, Mg = G.shape
    
    if weights is None:
        weights = np.ones(Mg)
    
    n_eq = A_eq.shape[0]
    
    # Build difference matrix D (for 1D TV)
    # For 2D grid, this is approximate; proper TV would need gradient on mesh
    D = np.zeros((Mg - 1, Mg))
    for i in range(Mg - 1):
        D[i, i] = -1
        D[i, i + 1] = 1
    
    # Edge weights: average of endpoint weights
    edge_weights = (weights[:-1] + weights[1:]) / 2
    
    # Initialize
    q = np.zeros(Mg)
    z = np.zeros(Mg - 1)  # Dq = z
    y = np.zeros(Mg - 1)
    lam = np.zeros(n_eq)
    
    # Precompute
    GtG = G.T @ G
    Gtu = G.T @ u
    DtD = D.T @ D
    
    # KKT for q-update: (G'G + rho*D'D) q + A_eq' lam = G'u + rho*D'(z - y)
    KKT_static = np.zeros((Mg + n_eq, Mg + n_eq))
    KKT_static[:Mg, :Mg] = GtG + rho_admm * DtD
    KKT_static[:Mg, Mg:] = A_eq.T
    KKT_static[Mg:, :Mg] = A_eq
    
    try:
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(KKT_static)
        use_lu = True
    except:
        use_lu = False
    
    for it in range(max_iter):
        q_old = q.copy()
        
        # q-update
        rhs = np.zeros(Mg + n_eq)
        rhs[:Mg] = Gtu + rho_admm * D.T @ (z - y)
        rhs[Mg:] = b_eq
        
        if use_lu:
            sol = lu_solve((lu, piv), rhs)
        else:
            sol = np.linalg.solve(KKT_static, rhs)
        
        q = sol[:Mg]
        
        # z-update: weighted soft thresholding
        v = D @ q + y
        threshold = alpha * edge_weights / rho_admm
        z = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        
        # y-update
        y = y + D @ q - z
        
        # Convergence check
        primal_res = np.linalg.norm(D @ q - z)
        dual_res = rho_admm * np.linalg.norm(D.T @ (z - (D @ q_old + y - (D @ q + y))))
        
        if primal_res < tol:
            break
    
    return q


# =============================================================================
# Alpha Selection
# =============================================================================

def compute_l_curve_weighted(G, u, reg_type='l2', weights=None, 
                              noise_level=None, n_alphas=50,
                              alpha_range=(1e-8, 1e1)):
    """
    Select regularization parameter via L-curve or discrepancy principle.
    
    If noise_level is provided, uses discrepancy principle:
        Select smallest alpha where ||Gq - u|| >= noise_level * sqrt(M) * 1.3
    
    Otherwise, uses L-curve corner detection.
    
    Parameters
    ----------
    G : ndarray (M, Mg)
        Forward matrix
    u : ndarray (M,)
        Measurement data
    reg_type : str
        'l2', 'l1', or 'tv'
    weights : ndarray (Mg,) or None
        Depth weights
    noise_level : float or None
        Noise standard deviation (for discrepancy principle)
    n_alphas : int
        Number of alpha values to test
    alpha_range : tuple
        (min_alpha, max_alpha)
    
    Returns
    -------
    alpha_opt : float
        Selected regularization parameter
    alphas : ndarray
        All tested alpha values
    metrics : dict
        Residuals and regularization values for each alpha
    """
    M, Mg = G.shape
    
    A_eq = np.ones((1, Mg))
    b_eq = np.array([0.0])
    
    alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), n_alphas)
    
    residuals = []
    reg_norms = []
    
    for alpha in alphas:
        try:
            if reg_type == 'l2':
                q = solve_l2_weighted(G, u, alpha, A_eq, b_eq, weights)
            elif reg_type == 'l1':
                q = solve_l1_weighted_admm(G, u, alpha, A_eq, b_eq, weights, max_iter=500)
            else:  # tv
                q = solve_tv_weighted(G, u, alpha, A_eq, b_eq, weights, max_iter=500)
            
            res = np.linalg.norm(G @ q - u)
            
            if reg_type == 'l2':
                if weights is not None:
                    reg = np.sqrt(np.sum(weights * q**2))
                else:
                    reg = np.linalg.norm(q)
            elif reg_type == 'l1':
                if weights is not None:
                    reg = np.sum(weights * np.abs(q))
                else:
                    reg = np.sum(np.abs(q))
            else:  # tv
                reg = np.sum(np.abs(np.diff(q)))
            
            residuals.append(res)
            reg_norms.append(reg)
        except:
            residuals.append(np.inf)
            reg_norms.append(np.inf)
    
    residuals = np.array(residuals)
    reg_norms = np.array(reg_norms)
    
    metrics = {'residuals': residuals, 'reg_norms': reg_norms}
    
    # Remove invalid entries
    valid = np.isfinite(residuals) & np.isfinite(reg_norms) & (reg_norms > 0)
    
    if not np.any(valid):
        warnings.warn("No valid alpha values found, returning middle of range")
        return alphas[n_alphas // 2], alphas, metrics
    
    # Discrepancy principle
    if noise_level is not None:
        target_residual = noise_level * np.sqrt(M) * 1.3  # Safety factor
        
        # Find smallest alpha where residual >= target
        for i, alpha in enumerate(alphas):
            if valid[i] and residuals[i] >= target_residual:
                return alpha, alphas, metrics
        
        # If none found, use largest alpha
        return alphas[valid][-1], alphas, metrics
    
    # L-curve corner detection
    log_res = np.log10(residuals[valid])
    log_reg = np.log10(reg_norms[valid])
    alphas_valid = alphas[valid]
    
    # Compute curvature
    if len(log_res) < 5:
        # Not enough points, return middle
        idx = len(alphas_valid) // 2
        return alphas_valid[idx], alphas, metrics
    
    # Numerical second derivative (curvature approximation)
    d1_res = np.gradient(log_res)
    d1_reg = np.gradient(log_reg)
    d2_res = np.gradient(d1_res)
    d2_reg = np.gradient(d1_reg)
    
    # Curvature formula
    curvature = np.abs(d1_res * d2_reg - d1_reg * d2_res) / (d1_res**2 + d1_reg**2)**1.5
    
    # Find maximum curvature
    idx = np.argmax(curvature[2:-2]) + 2  # Avoid edge effects
    
    return alphas_valid[idx], alphas, metrics


# =============================================================================
# Analysis Utilities
# =============================================================================

def compute_intensity_distribution(q, conformal_radii, zones=None):
    """
    Compute intensity distribution across radial zones.
    
    Parameters
    ----------
    q : ndarray
        Intensity vector
    conformal_radii : ndarray
        Conformal radius for each grid point
    zones : list of tuples or None
        Zone boundaries [(r_lo, r_hi), ...]. Default: 4 equal zones.
    
    Returns
    -------
    distribution : dict
        Intensity fraction in each zone
    """
    if zones is None:
        zones = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    
    total = np.sum(np.abs(q))
    
    distribution = {}
    for (r_lo, r_hi) in zones:
        mask = (conformal_radii >= r_lo) & (conformal_radii < r_hi)
        zone_intensity = np.sum(np.abs(q[mask]))
        zone_frac = zone_intensity / total if total > 0 else 0
        distribution[f'[{r_lo},{r_hi})'] = {
            'intensity': zone_intensity,
            'fraction': zone_frac,
            'n_points': np.sum(mask)
        }
    
    return distribution


def analyze_solver_bias(q, conformal_radii, true_rho_range):
    """
    Analyze whether solution has boundary bias.
    
    Parameters
    ----------
    q : ndarray
        Intensity vector
    conformal_radii : ndarray
        Conformal radii
    true_rho_range : tuple
        (rho_min, rho_max) where true sources are
    
    Returns
    -------
    metrics : dict
        Bias analysis metrics
    """
    total = np.sum(np.abs(q))
    
    # Target zone (where true sources are)
    target_mask = (conformal_radii >= true_rho_range[0] - 0.05) & \
                  (conformal_radii <= true_rho_range[1] + 0.05)
    target_intensity = np.sum(np.abs(q[target_mask]))
    
    # Boundary zone
    boundary_mask = conformal_radii > 0.9
    boundary_intensity = np.sum(np.abs(q[boundary_mask]))
    
    # Center zone
    center_mask = conformal_radii < 0.3
    center_intensity = np.sum(np.abs(q[center_mask]))
    
    return {
        'target_fraction': target_intensity / total if total > 0 else 0,
        'boundary_fraction': boundary_intensity / total if total > 0 else 0,
        'center_fraction': center_intensity / total if total > 0 else 0,
        'total_intensity': total
    }
