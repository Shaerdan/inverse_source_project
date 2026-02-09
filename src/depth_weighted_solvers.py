#!/usr/bin/env python3
"""
Depth-Weighted Linear Solvers for Inverse Source Problems

This module implements depth-weighted regularization to correct for the
fundamental depth sensitivity bias in linear inverse source methods.

The Problem:
-----------
The Green's function has higher sensitivity to boundary sources than interior sources.
Column norms of G vary by ~20x from center to boundary. Standard L2 regularization
(min ||q||²) therefore prefers boundary solutions that achieve the same data fit
with smaller intensity norm.

The Solution:
------------
Apply depth-dependent weights to the regularization term:
    min ||Gq - u||² + α ||Wq||²  (L2)
    min ||Gq - u||² + α Σ w_j |q_j|  (L1)
    min ||Gq - u||² + α Σ w_ij |q_i - q_j|  (TV)

where W = diag(w_j) with w_j = (1 - ρ_j)^(-β) penalizes boundary sources more heavily.

Parameters:
----------
β (beta): Distance weighting exponent
    - β = 0: No weighting (original behavior)
    - β = 1: Linear weighting (recommended)
    - β = 2: Quadratic weighting (stronger correction)

α (alpha): Regularization strength (selected via L-curve)

References:
----------
See depth_sensitivity_weighting.md for mathematical derivation.
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Optional


def compute_conformal_radii_disk(grid: np.ndarray) -> np.ndarray:
    """
    Compute conformal radii for disk domain.
    
    For the unit disk, conformal radius = Euclidean distance from origin.
    
    Parameters
    ----------
    grid : ndarray (N, 2)
        Grid points in physical coordinates
        
    Returns
    -------
    radii : ndarray (N,)
        Conformal radius for each grid point (0 = center, 1 = boundary)
    """
    return np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)


def compute_conformal_radii_general(grid: np.ndarray, cmap) -> np.ndarray:
    """
    Compute conformal radii for general domain via conformal map.
    
    Parameters
    ----------
    grid : ndarray (N, 2)
        Grid points in physical coordinates
    cmap : MFSConformalMap
        Conformal map from physical domain to unit disk
        
    Returns
    -------
    radii : ndarray (N,)
        Conformal radius |f(z)| for each grid point
    """
    radii = np.zeros(len(grid))
    for j in range(len(grid)):
        z_j = complex(grid[j, 0], grid[j, 1])
        w_j = cmap.to_disk(z_j)
        radii[j] = abs(w_j)
    return radii


def compute_depth_weights(radii: np.ndarray, beta: float = 1.0, 
                          max_radius: float = 0.99) -> np.ndarray:
    """
    Compute depth-dependent regularization weights.
    
    Weight formula: w_j = (1 - ρ_j)^(-β)
    
    This penalizes boundary sources (ρ → 1) more heavily than interior sources,
    counteracting the natural sensitivity bias of the Green's function.
    
    Parameters
    ----------
    radii : ndarray (N,)
        Conformal radii of grid points
    beta : float
        Weighting exponent (default 1.0)
        - β = 0: No weighting
        - β = 1: Linear weighting (recommended)
        - β = 2: Quadratic weighting
    max_radius : float
        Clip radii to avoid singularity at boundary (default 0.99)
        
    Returns
    -------
    weights : ndarray (N,)
        Regularization weights w_j = (1 - ρ_j)^(-β)
    """
    # Clip radii to avoid division by zero
    rho = np.clip(radii, 0, max_radius)
    
    # Compute weights
    if beta == 0:
        return np.ones(len(radii))
    else:
        return (1 - rho) ** (-beta)


def solve_l2_weighted(G: np.ndarray, u: np.ndarray, alpha: float,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Solve weighted L2-regularized problem with compatibility constraint.
    
    Minimizes: ||Gq - u||² + α ||Wq||²  subject to Σq = 0
    
    where W = diag(weights).
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    u : ndarray (M,)
        Measured boundary data
    alpha : float
        Regularization parameter
    weights : ndarray (N,), optional
        Depth weights. If None, uses uniform weights (standard L2).
        
    Returns
    -------
    q : ndarray (N,)
        Regularized solution satisfying Σq = 0
    """
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    
    # Build weighted regularization matrix
    if weights is None:
        W2 = np.eye(n)
    else:
        W2 = np.diag(weights ** 2)  # W^T W = diag(w²)
    
    # KKT system with constraint Σq = 0
    A = np.block([
        [GtG + alpha * W2, np.ones((n, 1))],
        [np.ones((1, n)), np.zeros((1, 1))]
    ])
    b = np.concatenate([Gtu, [0.0]])
    
    try:
        x = np.linalg.solve(A, b)
        return x[:n]
    except np.linalg.LinAlgError:
        return np.zeros(n)


def solve_l1_weighted_admm(G: np.ndarray, u: np.ndarray, alpha: float,
                           weights: Optional[np.ndarray] = None,
                           rho: float = 1.0, max_iter: int = 2000,
                           tol: float = 1e-7) -> np.ndarray:
    """
    Solve weighted L1-regularized problem via ADMM with compatibility constraint.
    
    Minimizes: ||Gq - u||² + α Σ w_j |q_j|  subject to Σq = 0
    
    ADMM formulation:
        min ||Gq - u||² + α ||Wz||₁  subject to q = z, Σq = 0
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    u : ndarray (M,)
        Measured boundary data
    alpha : float
        Regularization parameter
    weights : ndarray (N,), optional
        Depth weights. If None, uses uniform weights (standard L1).
    rho : float
        ADMM penalty parameter (default 1.0)
    max_iter : int
        Maximum iterations (default 2000)
    tol : float
        Convergence tolerance (default 1e-7)
        
    Returns
    -------
    q : ndarray (N,)
        Regularized solution satisfying Σq = 0
    """
    n = G.shape[1]
    if weights is None:
        weights = np.ones(n)
    
    GtG = G.T @ G
    Gtu = G.T @ u
    
    # Build KKT matrix for q-update (includes compatibility constraint)
    A_base = GtG + rho * np.eye(n)
    A = np.block([
        [A_base, np.ones((n, 1))],
        [np.ones((1, n)), np.zeros((1, 1))]
    ])
    
    # Initialize
    z = np.zeros(n)
    y = np.zeros(n)  # Dual variable
    
    for iteration in range(max_iter):
        # q-update: solve (G^TG + ρI)q + λ1 = G^Tu + ρ(z - y), 1^Tq = 0
        rhs = Gtu + rho * (z - y)
        b = np.concatenate([rhs, [0.0]])
        
        try:
            sol = np.linalg.solve(A, b)
            q = sol[:n]
        except np.linalg.LinAlgError:
            break
        
        # z-update: soft thresholding with weighted threshold
        v = q + y
        threshold = alpha * weights / rho
        z_new = np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
        
        # y-update (dual)
        y = y + q - z_new
        
        # Convergence check
        primal_res = np.linalg.norm(q - z_new)
        dual_res = rho * np.linalg.norm(z_new - z)
        z = z_new
        
        if primal_res < tol and dual_res < tol:
            break
    
    # Ensure exact compatibility
    q = q - np.mean(q)
    return q


def solve_tv_weighted(G: np.ndarray, u: np.ndarray, alpha: float,
                      edges: List[Tuple[int, int]],
                      weights: Optional[np.ndarray] = None,
                      max_iter: int = 50) -> np.ndarray:
    """
    Solve weighted TV-regularized problem via IRLS with compatibility constraint.
    
    Minimizes: ||Gq - u||² + α Σ_{(i,j)∈E} w_ij |q_i - q_j|  subject to Σq = 0
    
    where w_ij = (w_i + w_j) / 2 (average of endpoint weights).
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    u : ndarray (M,)
        Measured boundary data
    alpha : float
        Regularization parameter
    edges : list of (int, int)
        Edge list for TV graph (from Delaunay triangulation)
    weights : ndarray (N,), optional
        Depth weights. If None, uses uniform weights (standard TV).
    max_iter : int
        Maximum IRLS iterations (default 50)
        
    Returns
    -------
    q : ndarray (N,)
        Regularized solution satisfying Σq = 0
    """
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    
    if len(edges) == 0:
        return solve_l2_weighted(G, u, alpha, weights)
    
    if weights is None:
        weights = np.ones(n)
    
    # Precompute edge weights
    edge_weights = np.array([(weights[i] + weights[j]) / 2 for (i, j) in edges])
    
    q = np.zeros(n)
    eps = 1e-4
    
    for _ in range(max_iter):
        # Build weighted TV Laplacian-like matrix
        L = np.zeros((n, n))
        for k, (i, j) in enumerate(edges):
            diff = abs(q[i] - q[j]) + eps
            # Weight by both edge weight and inverse difference
            w = edge_weights[k] / diff
            L[i, i] += w
            L[j, j] += w
            L[i, j] -= w
            L[j, i] -= w
        
        # KKT system
        A = np.block([
            [GtG + alpha * L, np.ones((n, 1))],
            [np.ones((1, n)), np.zeros((1, 1))]
        ])
        b = np.concatenate([Gtu, [0.0]])
        
        try:
            x = np.linalg.solve(A, b)
            q_new = x[:n]
        except np.linalg.LinAlgError:
            break
        
        if np.linalg.norm(q_new - q) < 1e-6:
            break
        q = q_new
    
    return q


def compute_l_curve_weighted(G: np.ndarray, u: np.ndarray, alphas: np.ndarray,
                             method: str, weights: Optional[np.ndarray] = None,
                             edges: List = None, noise_level: float = None) -> Dict:
    """
    Compute L-curve and find optimal alpha for weighted regularization.
    
    Uses discrepancy principle when noise_level is provided: select smallest α
    such that residual ≈ noise_level * sqrt(M).
    Falls back to L-curve corner when noise_level is None.
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    u : ndarray (M,)
        Measured boundary data
    alphas : ndarray
        Array of regularization parameters to try
    method : str
        One of 'l2', 'l1', 'tv'
    weights : ndarray (N,), optional
        Depth weights
    edges : list, optional
        Edge list for TV method
    noise_level : float, optional
        Expected noise standard deviation (for discrepancy principle)
        
    Returns
    -------
    dict with keys:
        alphas : tested alpha values
        residuals : ||Gq - u|| for each alpha
        solution_norms : ||q|| for each alpha  
        solutions : list of solution vectors
        best_idx : index of optimal alpha
        best_alpha : optimal alpha value
        best_solution : solution at optimal alpha
        curvature : L-curve curvature values
    """
    residuals = []
    solution_norms = []
    solutions = []
    
    for alpha in alphas:
        if method == 'l2':
            q = solve_l2_weighted(G, u, alpha, weights)
        elif method == 'l1':
            q = solve_l1_weighted_admm(G, u, alpha, weights)
        elif method == 'tv':
            q = solve_tv_weighted(G, u, alpha, edges, weights)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        res = np.linalg.norm(G @ q - u)
        sol_norm = np.linalg.norm(q)
        
        residuals.append(res)
        solution_norms.append(sol_norm)
        solutions.append(q)
    
    residuals = np.array(residuals)
    solution_norms = np.array(solution_norms)
    
    # Alpha selection method
    if noise_level is not None:
        # Discrepancy principle: find smallest alpha where residual ≈ noise * sqrt(M)
        # Use factor of 1.2-1.5 to allow slightly more regularization
        target_residual = noise_level * np.sqrt(len(u)) * 1.3
        
        # Find first alpha (from small to large) where residual exceeds target
        valid_idx = np.where(residuals >= target_residual)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[0]
        else:
            # All residuals below target - use largest alpha
            best_idx = len(alphas) - 1
    else:
        # L-curve corner method
        log_res = np.log10(residuals + 1e-15)
        log_sol = np.log10(solution_norms + 1e-15)
        
        # Compute curvature
        dx = np.gradient(log_res)
        dy = np.gradient(log_sol)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-15)**1.5
        
        # Find maximum curvature (avoid endpoints)
        interior = slice(2, len(curvature) - 2)
        best_idx = np.argmax(curvature[interior]) + 2
    
    # Compute curvature for output (even if not used for selection)
    log_res = np.log10(residuals + 1e-15)
    log_sol = np.log10(solution_norms + 1e-15)
    dx = np.gradient(log_res)
    dy = np.gradient(log_sol)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-15)**1.5
    
    return {
        'alphas': alphas,
        'residuals': residuals,
        'solution_norms': solution_norms,
        'solutions': solutions,
        'best_idx': best_idx,
        'best_alpha': alphas[best_idx],
        'best_solution': solutions[best_idx],
        'curvature': curvature
    }


def select_beta_heuristic(G: np.ndarray, radii: np.ndarray) -> float:
    """
    Select depth weighting exponent β based on column norm sensitivity.
    
    Heuristic: β should counteract the sensitivity variation. If column norms
    vary by factor K from center to boundary, we need β such that the product
    of column norm and weight is approximately uniform.
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    radii : ndarray (N,)
        Conformal radii
        
    Returns
    -------
    beta : float
        Recommended weighting exponent
    """
    col_norms = np.linalg.norm(G, axis=0)
    
    # Fit: log(||G_j||) ≈ a + b * log(1/(1-ρ_j))
    # Then β ≈ b to counteract
    rho_clipped = np.clip(radii, 0.01, 0.99)
    log_sensitivity = np.log(1 / (1 - rho_clipped))
    log_norms = np.log(col_norms + 1e-10)
    
    # Linear regression
    A = np.column_stack([np.ones_like(log_sensitivity), log_sensitivity])
    try:
        coeffs = np.linalg.lstsq(A, log_norms, rcond=None)[0]
        beta = coeffs[1]
        # Clip to reasonable range
        return np.clip(beta, 0.5, 2.5)
    except:
        return 1.0  # Default


def run_weighted_linear_solver(G: np.ndarray, u: np.ndarray, 
                               radii: np.ndarray,
                               method: str = 'l1',
                               beta: Optional[float] = None,
                               alpha_range: np.ndarray = None,
                               edges: List = None) -> Dict:
    """
    Run depth-weighted linear solver with automatic parameter selection.
    
    Parameters
    ----------
    G : ndarray (M, N)
        Green's matrix
    u : ndarray (M,)
        Measured boundary data
    radii : ndarray (N,)
        Conformal radii of grid points
    method : str
        Regularization method: 'l2', 'l1', or 'tv'
    beta : float, optional
        Depth weighting exponent. If None, uses heuristic selection.
    alpha_range : ndarray, optional
        Range of alpha values for L-curve. Default: logspace(-8, -1, 30)
    edges : list, optional
        Edge list for TV method
        
    Returns
    -------
    dict with keys:
        solution : optimal solution vector
        alpha : selected regularization parameter
        beta : depth weighting exponent used
        weights : depth weights used
        residual : ||Gq - u||
        l_curve : full L-curve results
    """
    if alpha_range is None:
        alpha_range = np.logspace(-8, -1, 30)
    
    # Select beta
    if beta is None:
        beta = select_beta_heuristic(G, radii)
    
    # Compute weights
    weights = compute_depth_weights(radii, beta)
    
    # Run L-curve
    l_curve = compute_l_curve_weighted(G, u, alpha_range, method, weights, edges)
    
    q = l_curve['best_solution']
    
    return {
        'solution': q,
        'alpha': l_curve['best_alpha'],
        'beta': beta,
        'weights': weights,
        'residual': np.linalg.norm(G @ q - u),
        'l_curve': l_curve
    }


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def compute_intensity_distribution(q: np.ndarray, radii: np.ndarray,
                                   bins: List[Tuple[float, float]] = None) -> Dict:
    """
    Compute intensity distribution across radial bins.
    
    Parameters
    ----------
    q : ndarray (N,)
        Intensity vector
    radii : ndarray (N,)
        Conformal radii
    bins : list of (lo, hi), optional
        Radial bins. Default: [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        
    Returns
    -------
    dict mapping bin labels to percentage of total |q| in that bin
    """
    if bins is None:
        bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    
    total = np.sum(np.abs(q))
    if total < 1e-15:
        return {f'[{lo:.1f},{hi:.1f})': 0.0 for lo, hi in bins}
    
    result = {}
    for lo, hi in bins:
        mask = (radii >= lo) & (radii < hi)
        pct = 100 * np.sum(np.abs(q[mask])) / total
        result[f'[{lo:.1f},{hi:.1f})'] = pct
    
    return result


def analyze_solver_bias(q: np.ndarray, radii: np.ndarray, 
                        true_radii: np.ndarray = None) -> Dict:
    """
    Analyze whether solver has boundary bias.
    
    Parameters
    ----------
    q : ndarray (N,)
        Recovered intensity vector
    radii : ndarray (N,)
        Conformal radii of grid points
    true_radii : ndarray, optional
        True source radii for comparison
        
    Returns
    -------
    dict with bias metrics
    """
    # Find peaks (local maxima of |q|)
    q_abs = np.abs(q)
    threshold = 0.1 * np.max(q_abs)
    peak_mask = q_abs > threshold
    
    if np.sum(peak_mask) == 0:
        return {'mean_peak_radius': np.nan, 'boundary_fraction': np.nan}
    
    peak_radii = radii[peak_mask]
    peak_intensities = q_abs[peak_mask]
    
    # Intensity-weighted mean radius
    weighted_mean_radius = np.average(peak_radii, weights=peak_intensities)
    
    # Fraction of intensity at boundary (r > 0.8)
    boundary_fraction = np.sum(peak_intensities[peak_radii > 0.8]) / np.sum(peak_intensities)
    
    result = {
        'mean_peak_radius': weighted_mean_radius,
        'boundary_fraction': boundary_fraction,
        'n_peaks': np.sum(peak_mask)
    }
    
    if true_radii is not None:
        result['true_mean_radius'] = np.mean(true_radii)
        result['radius_bias'] = weighted_mean_radius - np.mean(true_radii)
    
    return result
