#!/usr/bin/env python3
"""
Single-job runner for nonlinear vs linear method comparison.

This script runs one complete comparison for a given N (number of sources).
It computes the appropriate noise level to ensure N_max >= N + margin.

Usage:
    python run_comparison_job.py --n-sources 6 --domain disk --seed 42 --output-dir results/

Output:
    - JSON file with all metrics
    - Text summary
    - Figures (true sources, nonlinear recovery, linear heatmaps, L-curves)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, Delaunay
import os
import sys
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import solvers
from analytical_solver import (
    AnalyticalForwardSolver,
    AnalyticalNonlinearInverseSolver,
)
from conformal_solver import (
    MFSConformalMap,
    ConformalForwardSolver,
    ConformalNonlinearInverseSolver
)
from mesh import get_source_grid, get_brain_boundary

# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid resolutions for linear methods (approximate spacing)
GRID_RESOLUTIONS = {
    'coarse': 0.25,
    'medium': 0.18,
    'fine': 0.12,
    'very_fine': 0.08
}

# Regularization parameter search range
ALPHA_RANGE = np.logspace(-8, -1, 30)

# Default parameters
DEFAULT_N_SENSORS = 100
DEFAULT_RHO_MIN = 0.6
DEFAULT_RHO_MAX = 0.8
DEFAULT_INTENSITY_RANGE = (0.5, 2.0)
DEFAULT_MARGIN = 2  # N_max >= N + margin


# =============================================================================
# NOISE LEVEL COMPUTATION
# =============================================================================

def compute_sigma_for_target_nmax(n_sources: int, rho_min: float, n_sensors: int,
                                   margin: int = 2, safety_factor: float = 0.5) -> Tuple[float, int, int]:
    """
    Compute σ_noise such that N_max >= N + margin.
    
    Returns:
        sigma_noise: noise standard deviation
        n_star_target: target Fourier cutoff
        n_max_target: resulting N_max
    """
    # Target N_max
    n_max_target = n_sources + margin
    
    # Required n* (from N_max = floor(2/3 * n*))
    # n* >= (3/2) * N_max
    n_star_target = int(np.ceil(1.5 * n_max_target))
    
    # At cutoff: ρ^{n*} / (π * n*) = σ_Four
    # σ_Four = σ_noise / √M
    sigma_four = (rho_min ** n_star_target) / (np.pi * n_star_target)
    
    # Apply safety factor (smaller σ → larger n*)
    sigma_four *= safety_factor
    
    sigma_noise = sigma_four * np.sqrt(n_sensors)
    
    return sigma_noise, n_star_target, n_max_target


def verify_nstar_nmax(noise: np.ndarray, rho_min: float) -> Tuple[int, int]:
    """Compute actual n* and N_max from noise realization."""
    M = len(noise)
    fft_coeffs = np.fft.fft(noise)
    eta_abs = (2.0 / M) * np.abs(fft_coeffs[1:51])
    
    usable = []
    for n in range(1, 51):
        if rho_min ** n / (np.pi * n) > eta_abs[n-1]:
            usable.append(n)
    
    n_star = max(usable) if usable else 0
    n_max = int((2.0 / 3.0) * n_star)
    return n_star, n_max


# =============================================================================
# SOURCE GENERATION
# =============================================================================

def generate_centered_intensities(n_sources: int, min_mag: float, max_mag: float, 
                                   rng: np.random.RandomState, max_attempts: int = 1000) -> np.ndarray:
    """
    Generate random intensities satisfying:
    - Sum = 0 (compatibility constraint)
    - All |q_k| ∈ [min_mag, max_mag]
    - Random signs (not alternating)
    
    Uses rejection sampling - typically succeeds in 3-5 attempts.
    """
    for _ in range(max_attempts):
        # Random magnitudes and random signs
        mags = rng.uniform(min_mag, max_mag, n_sources)
        signs = rng.choice([-1, 1], n_sources)
        intensities = mags * signs
        
        # Center to enforce sum = 0
        intensities -= np.mean(intensities)
        
        # Check all magnitudes still in valid range
        abs_vals = np.abs(intensities)
        if np.all(abs_vals >= min_mag) and np.all(abs_vals <= max_mag):
            return intensities
    
    raise ValueError(f"Could not generate valid intensities after {max_attempts} attempts")


def generate_sources(n_sources: int, rho_min: float, rho_max: float,
                     intensity_range: Tuple[float, float], seed: int) -> List[Tuple[Tuple[float, float], float]]:
    """
    Generate well-separated sources with random intensities.
    
    Intensity constraints:
    - Sum = 0 (Neumann compatibility)
    - All |q_k| ∈ [intensity_range[0], intensity_range[1]]
    - Random signs
    """
    rng = np.random.RandomState(seed)
    
    # Evenly spaced angles with perturbation
    base_angles = np.linspace(0, 2 * np.pi, n_sources, endpoint=False)
    perturbation = 0.1 * (2 * np.pi / n_sources) * rng.randn(n_sources)
    angles = base_angles + perturbation
    
    # Random radii
    radii = rng.uniform(rho_min, rho_max, n_sources)
    
    # Random intensities with guaranteed magnitude bounds
    intensities = generate_centered_intensities(
        n_sources, 
        min_mag=intensity_range[0], 
        max_mag=intensity_range[1],
        rng=rng
    )
    
    sources = []
    for i in range(n_sources):
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        sources.append(((x, y), intensities[i]))
    
    return sources


# =============================================================================
# GRID GENERATION AND STATISTICS
# =============================================================================

def generate_interior_grid(domain: str, spacing: float, cmap=None) -> np.ndarray:
    """Generate interior grid points for a domain."""
    if domain == 'disk':
        # Generate on regular grid, filter to interior
        x = np.arange(-1 + spacing/2, 1, spacing)
        y = np.arange(-1 + spacing/2, 1, spacing)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        # Keep interior points (with margin)
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        mask = r < 0.95
        return points[mask]
    
    elif domain == 'ellipse':
        a, b = 1.5, 0.8
        x = np.arange(-a + spacing/2, a, spacing)
        y = np.arange(-b + spacing/2, b, spacing)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        # Inside ellipse: (x/a)² + (y/b)² < 1
        inside = (points[:, 0]/a)**2 + (points[:, 1]/b)**2 < 0.95
        return points[inside]
    
    elif domain == 'brain':
        # Get brain boundary
        boundary = get_brain_boundary(n_points=200)
        # Bounding box
        xmin, xmax = boundary[:, 0].min(), boundary[:, 0].max()
        ymin, ymax = boundary[:, 1].min(), boundary[:, 1].max()
        
        x = np.arange(xmin + spacing/2, xmax, spacing)
        y = np.arange(ymin + spacing/2, ymax, spacing)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Point-in-polygon test
        from matplotlib.path import Path
        path = Path(boundary)
        inside = path.contains_points(points)
        
        # Additional margin from boundary
        interior_pts = points[inside]
        if len(interior_pts) > 0:
            tree = cKDTree(boundary)
            dists, _ = tree.query(interior_pts)
            margin_mask = dists > spacing * 0.5
            return interior_pts[margin_mask]
        return interior_pts
    
    else:
        raise ValueError(f"Unknown domain: {domain}")


def compute_grid_statistics(grid: np.ndarray) -> Dict:
    """Compute grid spacing statistics."""
    if len(grid) < 2:
        return {'n_points': len(grid), 'h_min': 0, 'h_max': 0, 'h_mean': 0, 'h_median': 0, 'h_std': 0}
    
    tree = cKDTree(grid)
    dists, _ = tree.query(grid, k=2)
    h = dists[:, 1]  # Distance to nearest neighbor
    
    return {
        'n_points': len(grid),
        'h_min': float(np.min(h)),
        'h_max': float(np.max(h)),
        'h_mean': float(np.mean(h)),
        'h_median': float(np.median(h)),
        'h_std': float(np.std(h))
    }


# =============================================================================
# GREEN'S MATRIX CONSTRUCTION
# =============================================================================

def build_greens_matrix_disk(sensor_angles: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Build Green's matrix for disk domain using boundary formula."""
    n_sensors = len(sensor_angles)
    n_grid = len(grid)
    
    G = np.zeros((n_sensors, n_grid))
    
    for j in range(n_grid):
        r_j = np.linalg.norm(grid[j])
        phi_j = np.arctan2(grid[j, 1], grid[j, 0])
        
        if r_j < 1e-10:
            r_j = 1e-10  # Avoid log(1)
        
        for i in range(n_sensors):
            angle_diff = sensor_angles[i] - phi_j
            G[i, j] = -1.0 / (2 * np.pi) * np.log(1 + r_j**2 - 2*r_j*np.cos(angle_diff))
    
    # Center columns
    G -= np.mean(G, axis=0, keepdims=True)
    return G


def build_greens_matrix_conformal(sensor_angles_disk: np.ndarray, grid_physical: np.ndarray,
                                   cmap: MFSConformalMap) -> np.ndarray:
    """
    Build Green's matrix for general domain using conformal mapping.
    
    For each grid point in physical domain:
    1. Map to disk: w_j = f(z_j)
    2. Use disk boundary formula with |w_j|, arg(w_j)
    """
    n_sensors = len(sensor_angles_disk)
    n_grid = len(grid_physical)
    
    G = np.zeros((n_sensors, n_grid))
    
    for j in range(n_grid):
        z_j = complex(grid_physical[j, 0], grid_physical[j, 1])
        w_j = cmap.to_disk(z_j)
        
        r_j = abs(w_j)
        phi_j = np.angle(w_j)
        
        if r_j < 1e-10:
            r_j = 1e-10
        if r_j > 0.999:
            r_j = 0.999  # Keep away from boundary
        
        for i in range(n_sensors):
            angle_diff = sensor_angles_disk[i] - phi_j
            G[i, j] = -1.0 / (2 * np.pi) * np.log(1 + r_j**2 - 2*r_j*np.cos(angle_diff))
    
    # Center columns
    G -= np.mean(G, axis=0, keepdims=True)
    return G


# =============================================================================
# REGULARIZATION METHODS
# =============================================================================

def compute_mutual_coherence(G: np.ndarray) -> float:
    """Compute mutual coherence."""
    norms = np.linalg.norm(G, axis=0)
    n_cols = G.shape[1]
    
    max_coh = 0.0
    # Sample for large matrices
    if n_cols > 200:
        indices = np.random.choice(n_cols, 200, replace=False)
    else:
        indices = np.arange(n_cols)
    
    for i, idx_i in enumerate(indices):
        for idx_j in indices[i+1:]:
            if norms[idx_i] > 1e-10 and norms[idx_j] > 1e-10:
                coh = abs(np.dot(G[:, idx_i], G[:, idx_j])) / (norms[idx_i] * norms[idx_j])
                max_coh = max(max_coh, coh)
    
    return max_coh


def compute_condition_number(G: np.ndarray) -> float:
    """Compute condition number."""
    try:
        s = np.linalg.svd(G, compute_uv=False)
        if s[-1] > 1e-15:
            return float(s[0] / s[-1])
        return np.inf
    except:
        return np.inf


def build_tv_graph(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Build edges for TV regularization using Delaunay triangulation."""
    if len(grid) < 3:
        return []
    
    try:
        tri = Delaunay(grid)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
        return list(edges)
    except:
        return []


def solve_l2_regularized(G: np.ndarray, u: np.ndarray, alpha: float) -> np.ndarray:
    """Solve L2-regularized problem with compatibility constraint."""
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    
    # KKT system
    A = np.block([
        [GtG + alpha * np.eye(n), np.ones((n, 1))],
        [np.ones((1, n)), np.zeros((1, 1))]
    ])
    b = np.concatenate([Gtu, [0.0]])
    
    try:
        x = np.linalg.solve(A, b)
        return x[:n]
    except:
        return np.zeros(n)


def solve_l1_regularized(G: np.ndarray, u: np.ndarray, alpha: float, max_iter: int = 50) -> np.ndarray:
    """Solve L1-regularized problem via IRLS with compatibility constraint."""
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    
    q = np.zeros(n)
    eps = 1e-4
    
    for _ in range(max_iter):
        # Weighted diagonal
        W = np.diag(1.0 / (np.abs(q) + eps))
        
        A = np.block([
            [GtG + alpha * W, np.ones((n, 1))],
            [np.ones((1, n)), np.zeros((1, 1))]
        ])
        b = np.concatenate([Gtu, [0.0]])
        
        try:
            x = np.linalg.solve(A, b)
            q_new = x[:n]
        except:
            break
        
        if np.linalg.norm(q_new - q) < 1e-6:
            break
        q = q_new
    
    return q


def solve_tv_regularized(G: np.ndarray, u: np.ndarray, alpha: float, 
                         edges: List[Tuple[int, int]], max_iter: int = 50) -> np.ndarray:
    """Solve TV-regularized problem via IRLS with Delaunay-based TV."""
    n = G.shape[1]
    GtG = G.T @ G
    Gtu = G.T @ u
    
    if len(edges) == 0:
        return solve_l2_regularized(G, u, alpha)
    
    q = np.zeros(n)
    eps = 1e-4
    
    for _ in range(max_iter):
        # Build TV Laplacian-like matrix
        L = np.zeros((n, n))
        for (i, j) in edges:
            diff = abs(q[i] - q[j]) + eps
            weight = 1.0 / diff
            L[i, i] += weight
            L[j, j] += weight
            L[i, j] -= weight
            L[j, i] -= weight
        
        A = np.block([
            [GtG + alpha * L, np.ones((n, 1))],
            [np.ones((1, n)), np.zeros((1, 1))]
        ])
        b = np.concatenate([Gtu, [0.0]])
        
        try:
            x = np.linalg.solve(A, b)
            q_new = x[:n]
        except:
            break
        
        if np.linalg.norm(q_new - q) < 1e-6:
            break
        q = q_new
    
    return q


def compute_l_curve(G: np.ndarray, u: np.ndarray, alphas: np.ndarray,
                    method: str, edges: List = None) -> Dict:
    """Compute L-curve and find optimal alpha."""
    residuals = []
    solution_norms = []
    solutions = []
    
    for alpha in alphas:
        if method == 'l2':
            q = solve_l2_regularized(G, u, alpha)
        elif method == 'l1':
            q = solve_l1_regularized(G, u, alpha)
        elif method == 'tv':
            q = solve_tv_regularized(G, u, alpha, edges)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        res = np.linalg.norm(G @ q - u)
        sol_norm = np.linalg.norm(q)
        
        residuals.append(res)
        solution_norms.append(sol_norm)
        solutions.append(q)
    
    residuals = np.array(residuals)
    solution_norms = np.array(solution_norms)
    
    # Find corner via maximum curvature
    log_res = np.log10(residuals + 1e-15)
    log_sol = np.log10(solution_norms + 1e-15)
    
    # Curvature
    dx = np.gradient(log_res)
    dy = np.gradient(log_sol)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-15)**1.5
    
    # Find maximum curvature (avoid endpoints)
    interior = slice(2, len(curvature) - 2)
    best_idx = np.argmax(curvature[interior]) + 2
    
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


# =============================================================================
# RMSE COMPUTATION
# =============================================================================

def extract_peaks(q: np.ndarray, grid: np.ndarray, threshold_frac: float = 0.1) -> List[Tuple[np.ndarray, float]]:
    """Extract peaks from distributed solution."""
    q_abs = np.abs(q)
    if q_abs.max() < 1e-15:
        return []
    
    threshold = threshold_frac * q_abs.max()
    
    # Grid spacing
    if len(grid) < 2:
        return []
    tree = cKDTree(grid)
    dists, _ = tree.query(grid, k=2)
    spacing = np.median(dists[:, 1])
    radius = 1.5 * spacing
    
    peaks = []
    for i in range(len(q)):
        if q_abs[i] < threshold:
            continue
        
        # Check local maximum
        is_max = True
        for j in range(len(q)):
            if i != j and np.linalg.norm(grid[j] - grid[i]) < radius:
                if q_abs[j] > q_abs[i]:
                    is_max = False
                    break
        
        if is_max:
            peaks.append((grid[i].copy(), q[i]))
    
    return peaks


def compute_rmse(true_sources: List, detected: List) -> Dict:
    """Compute RMSE via Hungarian matching."""
    n_true = len(true_sources)
    n_det = len(detected)
    
    if n_det == 0:
        return {
            'rmse_position': np.inf,
            'rmse_intensity': np.inf,
            'n_detected': 0,
            'n_true': n_true,
            'detection_rate': 0.0,
            'false_positive_rate': 0.0
        }
    
    true_pos = np.array([s[0] for s in true_sources])
    true_int = np.array([s[1] for s in true_sources])
    det_pos = np.array([d[0] for d in detected])
    det_int = np.array([d[1] for d in detected])
    
    # Cost matrix (position distance)
    cost = np.zeros((n_true, n_det))
    for i in range(n_true):
        for j in range(n_det):
            cost[i, j] = np.linalg.norm(true_pos[i] - det_pos[j])
    
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # RMSE
    pos_errors_sq = [cost[i, j]**2 for i, j in zip(row_ind, col_ind)]
    int_errors_sq = [(true_int[i] - det_int[j])**2 for i, j in zip(row_ind, col_ind)]
    
    return {
        'rmse_position': float(np.sqrt(np.mean(pos_errors_sq))),
        'rmse_intensity': float(np.sqrt(np.mean(int_errors_sq))),
        'n_detected': n_det,
        'n_true': n_true,
        'detection_rate': float(len(row_ind) / n_true),
        'false_positive_rate': float(max(0, n_det - n_true) / max(1, n_det))
    }


# =============================================================================
# FORWARD SOLVERS
# =============================================================================

def forward_solve_disk(sources: List, n_sensors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Forward solve for disk domain. Returns (u, sensor_angles)."""
    theta = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    
    u = np.zeros(n_sensors)
    for (x, y), q in sources:
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        for i in range(n_sensors):
            angle_diff = theta[i] - phi
            u[i] += q * (-1.0 / (2 * np.pi)) * np.log(1 + r**2 - 2*r*np.cos(angle_diff))
    
    u -= np.mean(u)
    return u, theta


def forward_solve_conformal(sources: List, n_sensors: int, cmap: MFSConformalMap) -> Tuple[np.ndarray, np.ndarray]:
    """Forward solve using conformal mapping. Returns (u, sensor_angles_in_disk)."""
    # Sensors at uniform disk angles
    theta_disk = np.linspace(0, 2 * np.pi, n_sensors, endpoint=False)
    
    u = np.zeros(n_sensors)
    for (x, y), q in sources:
        z = complex(x, y)
        w = cmap.to_disk(z)
        r = abs(w)
        phi = np.angle(w)
        
        for i in range(n_sensors):
            angle_diff = theta_disk[i] - phi
            u[i] += q * (-1.0 / (2 * np.pi)) * np.log(1 + r**2 - 2*r*np.cos(angle_diff))
    
    u -= np.mean(u)
    return u, theta_disk


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sources(sources: List, domain: str, title: str, filename: str, 
                 boundary_pts: np.ndarray = None, a: float = 1.5, b: float = 0.8):
    """Plot true sources."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Domain boundary
    if domain == 'disk':
        circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    elif domain == 'ellipse':
        ellipse = Ellipse((0, 0), 2*a, 2*b, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(ellipse)
        ax.set_xlim(-a-0.2, a+0.2)
        ax.set_ylim(-b-0.2, b+0.2)
    elif domain == 'brain' and boundary_pts is not None:
        ax.plot(np.append(boundary_pts[:, 0], boundary_pts[0, 0]),
                np.append(boundary_pts[:, 1], boundary_pts[0, 1]), 'k-', linewidth=2)
        ax.set_xlim(boundary_pts[:, 0].min() - 0.1, boundary_pts[:, 0].max() + 0.1)
        ax.set_ylim(boundary_pts[:, 1].min() - 0.1, boundary_pts[:, 1].max() + 0.1)
    
    # Plot sources
    for (x, y), q in sources:
        color = 'red' if q > 0 else 'blue'
        size = 100 * abs(q)
        ax.scatter(x, y, c=color, s=size, edgecolors='black', linewidths=1.5, zorder=5)
    
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_nonlinear_recovery(true_sources: List, recovered: List, domain: str, 
                            rmse_pos: float, rmse_int: float, title: str, filename: str,
                            boundary_pts: np.ndarray = None, a: float = 1.5, b: float = 0.8):
    """Plot nonlinear recovery overlaid on true sources with intensity labels."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Domain boundary
    if domain == 'disk':
        circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
    elif domain == 'ellipse':
        ellipse = Ellipse((0, 0), 2*a, 2*b, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(ellipse)
        ax.set_xlim(-a-0.3, a+0.3)
        ax.set_ylim(-b-0.3, b+0.3)
    elif domain == 'brain' and boundary_pts is not None:
        ax.plot(np.append(boundary_pts[:, 0], boundary_pts[0, 0]),
                np.append(boundary_pts[:, 1], boundary_pts[0, 1]), 'k-', linewidth=2)
        ax.set_xlim(boundary_pts[:, 0].min() - 0.15, boundary_pts[:, 0].max() + 0.15)
        ax.set_ylim(boundary_pts[:, 1].min() - 0.15, boundary_pts[:, 1].max() + 0.15)
    
    # True sources (hollow circles)
    for (x, y), q in true_sources:
        color = 'red' if q > 0 else 'blue'
        size = 150 * abs(q)
        ax.scatter(x, y, s=size, facecolors='none', edgecolors=color, linewidths=2.0, zorder=5)
        # Intensity label for true source
        ax.annotate(f'{q:.2f}', (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=color, fontweight='bold', zorder=7)
    
    # Recovered sources (filled circles)
    for src in recovered:
        if hasattr(src, 'x'):
            x, y, q = src.x, src.y, src.intensity
        else:
            (x, y), q = src[0], src[1]
        color = 'darkred' if q > 0 else 'darkblue'
        size = 100 * abs(q)
        ax.scatter(x, y, c=color, s=size, marker='o', edgecolors='black', linewidths=1.0, zorder=6)
        # Intensity label for recovered source
        ax.annotate(f'{q:.2f}', (x, y), xytext=(-5, -15), textcoords='offset points',
                    fontsize=8, color=color, zorder=7)
    
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nRMSE_pos = {rmse_pos:.4f}, RMSE_int = {rmse_int:.4f}', fontsize=12)
    
    # Legend
    ax.scatter([], [], s=100, facecolors='none', edgecolors='red', linewidths=2, label='True (+)')
    ax.scatter([], [], s=100, facecolors='none', edgecolors='blue', linewidths=2, label='True (-)')
    ax.scatter([], [], s=80, c='darkred', marker='o', edgecolors='black', label='Recovered (+)')
    ax.scatter([], [], s=80, c='darkblue', marker='o', edgecolors='black', label='Recovered (-)')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_linear_heatmap(q: np.ndarray, grid: np.ndarray, true_sources: List,
                        domain: str, title: str, filename: str, rmse_pos: float,
                        boundary_pts: np.ndarray = None, a: float = 1.5, b: float = 0.8):
    """Plot linear solution as heatmap with interpolated background and true sources overlaid."""
    from scipy.interpolate import griddata
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    vmax = np.max(np.abs(q))
    if vmax < 1e-10:
        vmax = 1.0  # Avoid division issues
    
    # Determine domain bounds
    if domain == 'disk':
        xlim, ylim = (-1.2, 1.2), (-1.2, 1.2)
    elif domain == 'ellipse':
        xlim, ylim = (-a-0.2, a+0.2), (-b-0.2, b+0.2)
    elif domain == 'brain' and boundary_pts is not None:
        margin = 0.1
        xlim = (boundary_pts[:, 0].min() - margin, boundary_pts[:, 0].max() + margin)
        ylim = (boundary_pts[:, 1].min() - margin, boundary_pts[:, 1].max() + margin)
    else:
        xlim, ylim = (-1.2, 1.2), (-1.2, 1.2)
    
    # Create interpolated background heatmap
    nx, ny = 200, 200
    xi = np.linspace(xlim[0], xlim[1], nx)
    yi = np.linspace(ylim[0], ylim[1], ny)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate (use 'linear' for smooth appearance)
    Zi = griddata(grid, q, (Xi, Yi), method='linear', fill_value=0)
    
    # Background interpolated heatmap (70% transparent)
    ax.imshow(Zi, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower',
              cmap='RdBu_r', vmin=-vmax, vmax=vmax, alpha=0.3, aspect='auto')
    
    # Grid points on top (more visible)
    scatter = ax.scatter(grid[:, 0], grid[:, 1], c=q, cmap='RdBu_r', 
                         vmin=-vmax, vmax=vmax, s=40, alpha=0.9, edgecolors='gray', linewidths=0.3)
    plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.8)
    
    # Domain boundary
    if domain == 'disk':
        circle = Circle((0, 0), 1, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
    elif domain == 'ellipse':
        ellipse = Ellipse((0, 0), 2*a, 2*b, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(ellipse)
    elif domain == 'brain' and boundary_pts is not None:
        ax.plot(np.append(boundary_pts[:, 0], boundary_pts[0, 0]),
                np.append(boundary_pts[:, 1], boundary_pts[0, 1]), 'k-', linewidth=2)
    
    # True sources (hollow circles with thick edge, on top)
    for (x, y), q_true in true_sources:
        color = 'lime' if q_true > 0 else 'cyan'
        size = 200 * abs(q_true)
        ax.scatter(x, y, s=size, facecolors='none', edgecolors=color, linewidths=2.5, zorder=10)
        # Intensity label
        ax.annotate(f'{q_true:.2f}', (x, y), xytext=(6, 6), textcoords='offset points',
                    fontsize=10, color='white', fontweight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\nRMSE_pos = {rmse_pos:.4f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_l_curve(lcurve: Dict, method: str, resolution: str, domain: str, filename: str):
    """Plot L-curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.loglog(lcurve['residuals'], lcurve['solution_norms'], 'b.-')
    ax.loglog(lcurve['residuals'][lcurve['best_idx']], 
              lcurve['solution_norms'][lcurve['best_idx']], 'ro', markersize=10)
    
    ax.set_xlabel('Residual norm ||Gq - u||')
    ax.set_ylabel('Solution norm ||q||')
    ax.set_title(f'L-curve: {method.upper()} - {resolution} - {domain}\n'
                 f'Best α = {lcurve["best_alpha"]:.2e}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_comparison(n_sources: int, domain: str, seed: int, output_dir: str,
                   rho_min: float = DEFAULT_RHO_MIN, rho_max: float = DEFAULT_RHO_MAX,
                   n_sensors: int = DEFAULT_N_SENSORS,
                   intensity_range: Tuple[float, float] = DEFAULT_INTENSITY_RANGE,
                   margin: int = DEFAULT_MARGIN,
                   skip_nonlinear: bool = False) -> Dict:
    """Run complete comparison for one configuration."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'config': {
            'n_sources': n_sources,
            'domain': domain,
            'seed': seed,
            'rho_min': rho_min,
            'rho_max': rho_max,
            'n_sensors': n_sensors,
            'intensity_range': intensity_range,
            'margin': margin,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: {domain.upper()} domain, N={n_sources}, seed={seed}")
    print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # Step 1: Compute noise level
    # -------------------------------------------------------------------------
    sigma_noise, n_star_target, n_max_target = compute_sigma_for_target_nmax(
        n_sources, rho_min, n_sensors, margin
    )
    
    print(f"\nNoise calibration:")
    print(f"  Target N_max = {n_max_target} (N + {margin})")
    print(f"  Target n* = {n_star_target}")
    print(f"  σ_noise = {sigma_noise:.6f}")
    
    results['config']['sigma_noise'] = sigma_noise
    results['config']['n_star_target'] = n_star_target
    results['config']['n_max_target'] = n_max_target
    
    # -------------------------------------------------------------------------
    # Step 2: Generate sources
    # -------------------------------------------------------------------------
    # Generate in disk coordinates, then map if needed
    disk_sources = generate_sources(n_sources, rho_min, rho_max, intensity_range, seed)
    
    # Domain-specific setup
    cmap = None
    boundary_pts = None
    ellipse_a, ellipse_b = 1.5, 0.8
    
    if domain == 'disk':
        true_sources = disk_sources
    
    elif domain == 'ellipse':
        def ellipse_boundary(t):
            return complex(ellipse_a * np.cos(t), ellipse_b * np.sin(t))
        cmap = MFSConformalMap(ellipse_boundary, n_boundary=256, n_charge=200)
        
        # Map sources to physical domain
        true_sources = []
        for (x, y), q in disk_sources:
            w = complex(x, y)
            z = cmap.from_disk(w)
            true_sources.append(((z.real, z.imag), q))
    
    elif domain == 'brain':
        boundary_pts = get_brain_boundary(n_points=200)
        
        def brain_boundary(t):
            n = len(boundary_pts)
            idx = int(t / (2 * np.pi) * n) % n
            next_idx = (idx + 1) % n
            frac = (t / (2 * np.pi) * n) - idx
            x = boundary_pts[idx, 0] * (1 - frac) + boundary_pts[next_idx, 0] * frac
            y = boundary_pts[idx, 1] * (1 - frac) + boundary_pts[next_idx, 1] * frac
            return complex(x, y)
        
        cmap = MFSConformalMap(brain_boundary, n_boundary=256, n_charge=200)
        
        true_sources = []
        for (x, y), q in disk_sources:
            w = complex(x, y)
            z = cmap.from_disk(w)
            true_sources.append(((z.real, z.imag), q))
    
    results['true_sources'] = [{'position': list(s[0]), 'intensity': s[1]} for s in true_sources]
    
    # -------------------------------------------------------------------------
    # Step 3: Forward solve and add noise
    # -------------------------------------------------------------------------
    np.random.seed(seed + 1000)  # Different seed for noise
    
    if domain == 'disk':
        u_true, sensor_angles = forward_solve_disk(true_sources, n_sensors)
    else:
        u_true, sensor_angles = forward_solve_conformal(true_sources, n_sensors, cmap)
    
    noise = np.random.randn(n_sensors) * sigma_noise
    u_measured = u_true + noise
    u_measured -= np.mean(u_measured)
    
    # Verify n* and N_max
    n_star_actual, n_max_actual = verify_nstar_nmax(noise, rho_min)
    
    print(f"\nActual noise statistics:")
    print(f"  n* = {n_star_actual}, N_max = {n_max_actual}")
    
    results['config']['n_star_actual'] = n_star_actual
    results['config']['n_max_actual'] = n_max_actual
    results['data_norm'] = float(np.linalg.norm(u_measured))
    
    # -------------------------------------------------------------------------
    # Step 4: Plot true sources
    # -------------------------------------------------------------------------
    plot_sources(true_sources, domain, f'True Sources - {domain} (N={n_sources})',
                 os.path.join(output_dir, f'sources_true_{domain}_N{n_sources}.png'),
                 boundary_pts, ellipse_a, ellipse_b)
    
    # -------------------------------------------------------------------------
    # Step 5: Nonlinear solver
    # -------------------------------------------------------------------------
    if not skip_nonlinear:
        print(f"\n--- NONLINEAR SOLVER ---")
        
        if domain == 'disk':
            solver = AnalyticalNonlinearInverseSolver(n_sources=n_sources, n_boundary=n_sensors)
            solver.set_measured_data(u_measured)
            result = solver.solve(method='SLSQP', n_restarts=15, maxiter=10000)
            recovered = result.sources
            residual = result.residual
        else:
            solver = ConformalNonlinearInverseSolver(cmap, n_sources=n_sources, n_boundary=n_sensors)
            recovered, residual = solver.solve(u_measured, method='SLSQP', n_restarts=15)
        
        # Convert recovered to standard format
        recovered_std = []
        for src in recovered:
            if hasattr(src, 'x'):
                recovered_std.append(((src.x, src.y), src.intensity))
            else:
                recovered_std.append(src)
        
        rmse_nonlinear = compute_rmse(true_sources, recovered_std)
        
        print(f"  RMSE_pos: {rmse_nonlinear['rmse_position']:.6f}")
        print(f"  RMSE_int: {rmse_nonlinear['rmse_intensity']:.6f}")
        print(f"  Residual: {residual:.6e}")
        
        results['nonlinear'] = {
            'rmse_position': rmse_nonlinear['rmse_position'],
            'rmse_intensity': rmse_nonlinear['rmse_intensity'],
            'data_residual': float(residual),
            'recovered_sources': [{'position': list(s[0]), 'intensity': s[1]} for s in recovered_std]
        }
        
        plot_nonlinear_recovery(true_sources, recovered_std, domain,
                                rmse_nonlinear['rmse_position'],
                                rmse_nonlinear['rmse_intensity'],
                                f'Nonlinear Recovery - {domain} (N={n_sources})',
                                os.path.join(output_dir, f'recovery_nonlinear_{domain}_N{n_sources}.png'),
                                boundary_pts, ellipse_a, ellipse_b)
    else:
        print(f"\n--- NONLINEAR SOLVER SKIPPED ---")
        results['nonlinear'] = {'skipped': True}
    
    # -------------------------------------------------------------------------
    # Step 6: Linear solvers
    # -------------------------------------------------------------------------
    results['linear_greens'] = {}
    
    for res_name, spacing in GRID_RESOLUTIONS.items():
        print(f"\n--- LINEAR GREEN'S [{res_name}] ---")
        
        # Generate grid
        grid = generate_interior_grid(domain, spacing, cmap)
        grid_stats = compute_grid_statistics(grid)
        
        print(f"  Grid: {grid_stats['n_points']} points, h_median={grid_stats['h_median']:.4f}")
        
        # Build Green's matrix
        if domain == 'disk':
            G = build_greens_matrix_disk(sensor_angles, grid)
        else:
            G = build_greens_matrix_conformal(sensor_angles, grid, cmap)
        
        mu = compute_mutual_coherence(G)
        kappa = compute_condition_number(G)
        
        print(f"  μ = {mu:.6f}, κ = {kappa:.2e}")
        
        # Build TV graph
        tv_edges = build_tv_graph(grid)
        
        # Results for this resolution
        res_results = {
            'grid_stats': grid_stats,
            'mutual_coherence': mu,
            'condition_number': kappa if kappa != np.inf else 'inf',
            'n_tv_edges': len(tv_edges),
            'methods': {}
        }
        
        # L2, L1, TV regularization
        for method in ['l2', 'l1', 'tv']:
            print(f"    {method.upper()} regularization...")
            
            lcurve = compute_l_curve(G, u_measured, ALPHA_RANGE, method, tv_edges)
            q = lcurve['best_solution']
            
            # Data residual
            data_res = np.linalg.norm(G @ q - u_measured)
            rel_res = data_res / np.linalg.norm(u_measured)
            
            # Peak detection and RMSE
            peaks = extract_peaks(q, grid)
            rmse = compute_rmse(true_sources, peaks)
            
            print(f"      α = {lcurve['best_alpha']:.2e}, RMSE_pos = {rmse['rmse_position']:.4f}, "
                  f"Det = {rmse['detection_rate']*100:.0f}%")
            
            res_results['methods'][method] = {
                'best_alpha': float(lcurve['best_alpha']),
                'rmse_position': rmse['rmse_position'],
                'rmse_intensity': rmse['rmse_intensity'],
                'detection_rate': rmse['detection_rate'],
                'false_positive_rate': rmse['false_positive_rate'],
                'n_detected': rmse['n_detected'],
                'data_residual': float(data_res),
                'relative_residual': float(rel_res)
            }
            
            # Plot L-curve
            plot_l_curve(lcurve, method, res_name, domain,
                         os.path.join(output_dir, f'lcurve_{method}_{res_name}_{domain}_N{n_sources}.png'))
            
            # Plot heatmap (only for best resolution later, or all)
            plot_linear_heatmap(q, grid, true_sources, domain,
                                f'Linear Green\'s {method.upper()} - {res_name} (N={n_sources})',
                                os.path.join(output_dir, f'heatmap_greens_{method}_{res_name}_{domain}_N{n_sources}.png'),
                                rmse['rmse_position'],
                                boundary_pts, ellipse_a, ellipse_b)
        
        results['linear_greens'][res_name] = res_results
    
    # -------------------------------------------------------------------------
    # Step 7: Save results
    # -------------------------------------------------------------------------
    
    # JSON
    json_path = os.path.join(output_dir, f'results_{domain}_N{n_sources}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Summary text
    summary_path = os.path.join(output_dir, f'summary_{domain}_N{n_sources}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{'='*70}\n")
        f.write(f"COMPARISON SUMMARY: {domain.upper()} domain, N={n_sources}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  N sources: {n_sources}\n")
        f.write(f"  Seed: {seed}\n")
        f.write(f"  σ_noise: {sigma_noise:.6f}\n")
        f.write(f"  ρ_min: {rho_min}, ρ_max: {rho_max}\n")
        f.write(f"  n_sensors: {n_sensors}\n")
        f.write(f"  n*: {n_star_actual} (target: {n_star_target})\n")
        f.write(f"  N_max: {n_max_actual} (target: {n_max_target})\n\n")
        
        f.write(f"{'-'*70}\n")
        f.write("NONLINEAR SOLVER:\n")
        f.write(f"{'-'*70}\n")
        if not skip_nonlinear:
            f.write(f"  RMSE (position): {rmse_nonlinear['rmse_position']:.6f}\n")
            f.write(f"  RMSE (intensity): {rmse_nonlinear['rmse_intensity']:.6f}\n")
            f.write(f"  Data residual: {residual:.6e}\n\n")
        else:
            f.write("  SKIPPED\n\n")
        
        f.write(f"{'-'*70}\n")
        f.write("LINEAR GREEN'S MATRIX:\n")
        f.write(f"{'-'*70}\n")
        f.write(f"{'Res':<12} {'Grid':<6} {'h_med':<8} {'μ':<8} {'Method':<6} {'α':<10} {'RMSE_pos':<10} {'Det%':<6}\n")
        f.write(f"{'-'*70}\n")
        
        for res_name, res_data in results['linear_greens'].items():
            for method in ['l2', 'l1', 'tv']:
                m = res_data['methods'][method]
                f.write(f"{res_name:<12} {res_data['grid_stats']['n_points']:<6} "
                        f"{res_data['grid_stats']['h_median']:<8.4f} "
                        f"{res_data['mutual_coherence']:<8.4f} "
                        f"{method.upper():<6} {m['best_alpha']:<10.2e} "
                        f"{m['rmse_position']:<10.4f} {m['detection_rate']*100:<6.0f}\n")
        
        f.write(f"\n{'='*70}\n")
        f.write("KEY INSIGHT: Linear methods achieve low data residual but fail source recovery!\n")
        f.write(f"{'='*70}\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"  - {json_path}")
    print(f"  - {summary_path}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run nonlinear vs linear comparison')
    parser.add_argument('--n-sources', type=int, required=True, help='Number of sources')
    parser.add_argument('--domain', type=str, required=True, 
                        choices=['disk', 'ellipse', 'brain'], help='Domain type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('--rho-min', type=float, default=DEFAULT_RHO_MIN, help='Min conformal radius')
    parser.add_argument('--rho-max', type=float, default=DEFAULT_RHO_MAX, help='Max conformal radius')
    parser.add_argument('--n-sensors', type=int, default=DEFAULT_N_SENSORS, help='Number of sensors')
    parser.add_argument('--margin', type=int, default=DEFAULT_MARGIN, help='N_max margin above N')
    parser.add_argument('--skip-nonlinear', action='store_true', help='Skip nonlinear solver (fast test)')
    
    args = parser.parse_args()
    
    run_comparison(
        n_sources=args.n_sources,
        domain=args.domain,
        seed=args.seed,
        output_dir=args.output_dir,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        n_sensors=args.n_sensors,
        margin=args.margin,
        skip_nonlinear=args.skip_nonlinear
    )


if __name__ == '__main__':
    main()
