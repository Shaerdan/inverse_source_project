"""
Parameter Selection for Inverse Source Problem

Implements:
- L-curve analysis for optimal regularization parameter selection
- Discrepancy principle (Morozov)
- GCV (Generalized Cross-Validation)
- Parameter sweep with diagnostic plots

Author: Claude (Anthropic)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.spatial import Delaunay
import warnings


@dataclass
class ParameterSweepResult:
    """Results from parameter sweep for a single method."""
    method: str
    alphas: np.ndarray
    residuals: np.ndarray  # ||Gq - u||
    regularizers: np.ndarray  # ||q||_1, ||q||_2, or ||Dq||_1
    n_peaks: np.ndarray
    position_rmses: np.ndarray  # Only if ground truth provided (CAUTION: threshold-dependent)
    max_intensities: np.ndarray
    solutions: List[np.ndarray]  # Store all solutions
    
    # Better metrics (not threshold-dependent)
    localization_scores: np.ndarray  # How much intensity is near true sources
    sparsity_ratios: np.ndarray  # How concentrated the solution is
    
    # Optimal parameters
    alpha_lcurve: float  # L-curve corner
    alpha_discrepancy: float  # Morozov discrepancy
    alpha_best_rmse: float  # Best position RMSE (CAUTION: may overfit)
    alpha_best_localization: float  # Best localization score
    
    idx_lcurve: int
    idx_discrepancy: int
    idx_best_rmse: int
    idx_best_localization: int


# =============================================================================
# RECONSTRUCTION QUALITY METRICS
# =============================================================================

def localization_score(q_recon: np.ndarray, points: np.ndarray, 
                       sources_true: List[Tuple], sigma: float = None) -> float:
    """
    Compute how much reconstructed intensity is near true sources.
    
    This metric doesn't require thresholding or peak detection.
    
    Parameters
    ----------
    q_recon : array (n,)
        Reconstructed source intensities on grid
    points : array (n, 2)
        Grid point coordinates
    sources_true : list of ((x, y), intensity)
        True source locations and intensities
    sigma : float or None
        Gaussian width for "nearness". If None, auto-computed as 10% of domain size.
        
    Returns
    -------
    score : float
        Score between 0 and 1:
        - 1 = all intensity concentrated exactly at true sources
        - 0 = all intensity far from true sources
    """
    from scipy.spatial.distance import cdist
    
    # Auto-compute sigma based on domain size if not provided
    if sigma is None:
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        domain_size = max(x_range, y_range)
        sigma = 0.1 * domain_size  # 10% of domain size
    
    true_pos = np.array([s[0] for s in sources_true])
    true_int = np.array([s[1] for s in sources_true])
    
    scores = []
    for sign_mult in [1, -1]:
        # True sources of this sign
        mask_true = (true_int * sign_mult) > 0
        if not mask_true.any():
            continue
        pos_true = true_pos[mask_true]
        
        # Reconstructed intensity of this sign
        if sign_mult > 0:
            q_sign = np.maximum(q_recon, 0)
        else:
            q_sign = np.maximum(-q_recon, 0)
        
        total_recon = q_sign.sum()
        if total_recon < 1e-10:
            scores.append(0)
            continue
        
        # For each recon point, compute max Gaussian weight to any true source
        dists = cdist(points, pos_true)
        weights = np.exp(-dists**2 / (2 * sigma**2))
        max_weights = weights.max(axis=1)
        
        # Score = intensity-weighted average of weights
        score = np.sum(q_sign * max_weights) / total_recon
        scores.append(score)
    
    return np.mean(scores) if scores else 0


def sparsity_ratio(q_recon: np.ndarray, target_sources: int = 4) -> float:
    """
    Measure solution sparsity relative to expected number of sources.
    
    Parameters
    ----------
    q_recon : array (n,)
        Reconstructed source intensities
    target_sources : int
        Expected number of sources (default: 4)
        
    Returns
    -------
    ratio : float
        Ratio between 0 and 1:
        - 1 = 90% of intensity in exactly target_sources points
        - Lower values = intensity spread across many points
    """
    q_abs = np.abs(q_recon)
    total = q_abs.sum()
    if total < 1e-10:
        return 0
    
    # Sort descending
    sorted_q = np.sort(q_abs)[::-1]
    cumsum = np.cumsum(sorted_q)
    
    # How many points to get 90% of intensity?
    n_for_90 = np.searchsorted(cumsum, 0.9 * total) + 1
    
    # Ratio: target_sources points should contain 90% ideally
    return min(target_sources / n_for_90, 1.0)


def intensity_weighted_centroid(points: np.ndarray, intensities: np.ndarray, 
                                 sign: str = None) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute intensity-weighted centroid for sources of given sign.
    
    Parameters
    ----------
    points : array (n, 2)
        Point coordinates
    intensities : array (n,)
        Intensity values
    sign : str, optional
        'positive', 'negative', or None (all)
        
    Returns
    -------
    centroid : array (2,) or None
        Intensity-weighted centroid, or None if no intensity
    total_intensity : float
        Sum of absolute intensities
    """
    if sign == 'positive':
        mask = intensities > 0
    elif sign == 'negative':
        mask = intensities < 0
    else:
        mask = np.ones(len(intensities), dtype=bool)
    
    weights = np.abs(intensities[mask])
    if weights.sum() < 1e-10:
        return None, 0
    
    centroid = np.average(points[mask], weights=weights, axis=0)
    total_intensity = weights.sum()
    return centroid, total_intensity


# =============================================================================
# ALPHA ESTIMATION
# =============================================================================

def estimate_alpha(noise_level: float, method: str = 'l2') -> float:
    """
    Estimate regularization parameter based on noise level.
    
    Parameters
    ----------
    noise_level : float
        Standard deviation of measurement noise
    method : str
        'l1', 'l2', or 'tv'
        
    Returns
    -------
    alpha : float
        Recommended starting regularization parameter
        
    Notes
    -----
    These are rough estimates. Use parameter_sweep() for proper selection.
    
    Rules of thumb:
    - L2: α ≈ σ² (noise variance)
    - L1: α ≈ σ² (same as L2)
    - TV: α ≈ 100 × σ² (needs much larger value)
    """
    base_alpha = noise_level ** 2
    
    if method.lower() == 'tv':
        return base_alpha * 100
    elif method.lower() in ('l1', 'l2'):
        return base_alpha
    else:
        raise ValueError(f"Unknown method: {method}")


def find_lcurve_corner(residuals: np.ndarray, regularizers: np.ndarray) -> int:
    """
    Find the corner of the L-curve using maximum curvature.
    
    Parameters
    ----------
    residuals : array
        Residual norms ||Gq - u||
    regularizers : array
        Regularization norms (||q||, ||Dq||, etc.)
        
    Returns
    -------
    corner_idx : int
        Index of the L-curve corner point
    """
    # Work in log space
    x = np.log10(residuals + 1e-16)
    y = np.log10(regularizers + 1e-16)
    
    # Method 1: Maximum distance from line connecting endpoints
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    
    if line_len < 1e-10:
        return len(x) // 2
    
    line_unit = line_vec / line_len
    
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        proj = np.dot(p - p1, line_unit)
        closest = p1 + proj * line_unit
        dist = np.linalg.norm(p - closest)
        distances.append(dist)
    
    return int(np.argmax(distances))


def find_discrepancy_alpha(residuals: np.ndarray, alphas: np.ndarray,
                           noise_level: float, n_measurements: int,
                           tau: float = 1.0) -> int:
    """
    Find alpha using Morozov's discrepancy principle.
    
    Choose alpha such that ||Gq - u|| ≈ τ * σ * √n
    
    Parameters
    ----------
    residuals : array
        Residual norms for each alpha
    alphas : array
        Regularization parameters
    noise_level : float
        Noise standard deviation σ
    n_measurements : int
        Number of measurements
    tau : float
        Safety factor (typically 1.0 to 1.5)
        
    Returns
    -------
    idx : int
        Index of alpha satisfying discrepancy principle
    """
    target = tau * noise_level * np.sqrt(n_measurements)
    
    # Find alpha where residual is closest to target
    # (from below, preferring slight over-regularization)
    idx = np.argmin(np.abs(residuals - target))
    
    return idx


def build_gradient_operator(points: np.ndarray) -> np.ndarray:
    """
    Build gradient operator D for TV regularization on a point cloud.
    
    Uses Delaunay triangulation to find edges.
    
    Parameters
    ----------
    points : array (n, 2)
        Point coordinates
        
    Returns
    -------
    D : array (n_edges, n)
        Gradient operator matrix
    """
    tri = Delaunay(points)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            edges.add(tuple(sorted([s[i], s[(i+1) % 3]])))
    edges = list(edges)
    
    n = len(points)
    D = np.zeros((len(edges), n))
    for k, (i, j) in enumerate(edges):
        D[k, i] = 1
        D[k, j] = -1
    
    return D


def parameter_sweep(
    G: np.ndarray,
    u_measured: np.ndarray,
    interior_points: np.ndarray,
    method: str = 'l2',
    alphas: np.ndarray = None,
    noise_level: float = None,
    sources_true: List[Tuple] = None,
    D: np.ndarray = None,
    verbose: bool = True
) -> ParameterSweepResult:
    """
    Perform parameter sweep with L-curve analysis.
    
    Parameters
    ----------
    G : array (m, n)
        Green's function matrix
    u_measured : array (m,)
        Boundary measurements
    interior_points : array (n, 2)
        Source candidate positions
    method : str
        'l1', 'l2', or 'tv'
    alphas : array, optional
        Regularization parameters to test. Default: logspace(-7, -1, 30)
    noise_level : float, optional
        Noise level for discrepancy principle
    sources_true : list, optional
        True sources for computing position RMSE
    D : array, optional
        Gradient operator for TV (built automatically if not provided)
    verbose : bool
        Print progress
        
    Returns
    -------
    result : ParameterSweepResult
        Complete sweep results with optimal parameters
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("cvxpy required for parameter sweep. Install with: pip install cvxpy")
    
    # Default alpha range
    if alphas is None:
        alphas = np.logspace(-7, -1, 30)
    
    # Center measurements
    u = u_measured - np.mean(u_measured)
    n = G.shape[1]
    m = len(u)
    
    # Build gradient operator for TV if needed
    if method.lower() == 'tv' and D is None:
        D = build_gradient_operator(interior_points)
    
    # Storage
    residuals = []
    regularizers = []
    n_peaks_list = []
    position_rmses = []
    max_intensities = []
    solutions = []
    
    if verbose:
        print(f"Parameter sweep for {method.upper()} ({len(alphas)} values)...")
    
    for i, alpha in enumerate(alphas):
        if verbose and i % 10 == 0:
            print(f"  α = {alpha:.2e} ({i+1}/{len(alphas)})")
        
        # Solve optimization problem
        q_var = cp.Variable(n)
        constraints = [cp.sum(q_var) == 0]  # Compatibility
        
        if method.lower() == 'l1':
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u) + alpha * cp.norm1(q_var)
            )
            reg_norm = lambda q: np.sum(np.abs(q))
        elif method.lower() == 'l2':
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u) + 0.5 * alpha * cp.sum_squares(q_var)
            )
            reg_norm = lambda q: np.linalg.norm(q)
        elif method.lower() == 'tv':
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u) + alpha * cp.norm1(D @ q_var)
            )
            reg_norm = lambda q: np.sum(np.abs(D @ q))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(verbose=False)
            q = q_var.value if q_var.value is not None else np.zeros(n)
        except:
            q = np.zeros(n)
        
        # Compute metrics
        res = np.linalg.norm(G @ q - u)
        reg = reg_norm(q)
        
        residuals.append(res)
        regularizers.append(reg)
        max_intensities.append(np.max(np.abs(q)))
        solutions.append(q)
        
        # Peak detection
        try:
            from .comparison import find_intensity_peaks, compute_peak_metrics
        except ImportError:
            from comparison import find_intensity_peaks, compute_peak_metrics
        
        peaks = find_intensity_peaks(interior_points, q)
        n_peaks_list.append(len(peaks))
        
        # Position RMSE if ground truth available (CAUTION: threshold-dependent)
        if sources_true is not None:
            metrics = compute_peak_metrics(sources_true, peaks)
            position_rmses.append(metrics['position_rmse'])
        else:
            position_rmses.append(np.nan)
    
    # Convert to arrays
    residuals = np.array(residuals)
    regularizers = np.array(regularizers)
    n_peaks_arr = np.array(n_peaks_list)
    position_rmses = np.array(position_rmses)
    max_intensities = np.array(max_intensities)
    
    # Compute better metrics (not threshold-dependent)
    localization_scores = np.zeros(len(alphas))
    sparsity_ratios = np.zeros(len(alphas))
    
    if sources_true is not None:
        n_sources = len(sources_true)
        for i, q in enumerate(solutions):
            localization_scores[i] = localization_score(q, interior_points, sources_true)
            sparsity_ratios[i] = sparsity_ratio(q, target_sources=n_sources)
    else:
        for i, q in enumerate(solutions):
            sparsity_ratios[i] = sparsity_ratio(q, target_sources=4)
    
    # Find optimal parameters
    idx_lcurve = find_lcurve_corner(residuals, regularizers)
    
    if noise_level is not None:
        idx_discrepancy = find_discrepancy_alpha(residuals, alphas, noise_level, m)
    else:
        idx_discrepancy = idx_lcurve  # Fallback
    
    if sources_true is not None:
        idx_best_rmse = int(np.nanargmin(position_rmses))
        idx_best_localization = int(np.argmax(localization_scores))
    else:
        idx_best_rmse = idx_lcurve  # Fallback
        idx_best_localization = idx_lcurve
    
    result = ParameterSweepResult(
        method=method,
        alphas=alphas,
        residuals=residuals,
        regularizers=regularizers,
        n_peaks=n_peaks_arr,
        position_rmses=position_rmses,
        max_intensities=max_intensities,
        solutions=solutions,
        localization_scores=localization_scores,
        sparsity_ratios=sparsity_ratios,
        alpha_lcurve=alphas[idx_lcurve],
        alpha_discrepancy=alphas[idx_discrepancy],
        alpha_best_rmse=alphas[idx_best_rmse],
        alpha_best_localization=alphas[idx_best_localization],
        idx_lcurve=idx_lcurve,
        idx_discrepancy=idx_discrepancy,
        idx_best_rmse=idx_best_rmse,
        idx_best_localization=idx_best_localization,
    )
    
    if verbose:
        print(f"\nOptimal parameters for {method.upper()}:")
        print(f"  L-curve corner:     α = {result.alpha_lcurve:.2e}")
        print(f"  Discrepancy:        α = {result.alpha_discrepancy:.2e}")
        if sources_true is not None:
            print(f"  Best localization:  α = {result.alpha_best_localization:.2e} (score = {localization_scores[idx_best_localization]:.4f})")
            print(f"  Best RMSE:          α = {result.alpha_best_rmse:.2e} (RMSE = {position_rmses[idx_best_rmse]:.4f}) [CAUTION: may overfit]")
    
    return result


def plot_parameter_sweep(
    results: Dict[str, ParameterSweepResult],
    save_path: str = None,
    show: bool = True
):
    """
    Plot L-curve and RMSE curves for all methods.
    
    Parameters
    ----------
    results : dict
        {method: ParameterSweepResult} for each method
    save_path : str, optional
        Path to save figure
    show : bool
        Display the figure
    """
    import matplotlib.pyplot as plt
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 12))
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    colors = {'l1': 'C0', 'l2': 'C1', 'tv': 'C2'}
    titles = {'l1': 'L1 (Lasso)', 'l2': 'L2 (Tikhonov)', 'tv': 'TV (Total Variation)'}
    
    for col, method in enumerate(methods):
        r = results[method]
        color = colors.get(method, 'C0')
        title = titles.get(method, method.upper())
        
        # Row 1: L-curve
        ax = axes[0, col]
        x = np.log10(r.residuals + 1e-16)
        y = np.log10(r.regularizers + 1e-16)
        sc = ax.scatter(x, y, c=np.log10(r.alphas), cmap='viridis', s=50)
        ax.plot(x, y, 'k-', alpha=0.3)
        ax.scatter([x[r.idx_lcurve]], [y[r.idx_lcurve]], c='red', s=200, marker='*',
                   label=f'L-curve: α={r.alpha_lcurve:.1e}', zorder=5)
        ax.scatter([x[r.idx_discrepancy]], [y[r.idx_discrepancy]], c='green', s=150, marker='s',
                   label=f'Discrepancy: α={r.alpha_discrepancy:.1e}', zorder=5)
        plt.colorbar(sc, ax=ax, label='log10(α)')
        ax.set_xlabel('log10(||Gq - u||)')
        ax.set_ylabel('log10(regularizer)')
        ax.set_title(f'{title}\nL-curve')
        ax.legend(fontsize=8)
        
        # Row 2: Position RMSE vs alpha
        ax = axes[1, col]
        if not np.all(np.isnan(r.position_rmses)):
            ax.semilogx(r.alphas, r.position_rmses, 'o-', color=color)
            ax.axvline(r.alpha_lcurve, color='red', linestyle='--', alpha=0.7, label='L-curve')
            ax.axvline(r.alpha_discrepancy, color='green', linestyle=':', alpha=0.7, label='Discrepancy')
            ax.scatter([r.alpha_best_rmse], [r.position_rmses[r.idx_best_rmse]], 
                      c='purple', s=200, marker='*', zorder=5,
                      label=f'Best: {r.position_rmses[r.idx_best_rmse]:.3f}')
            ax.set_ylabel('Position RMSE')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No ground truth', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('α')
        ax.set_title('Position RMSE vs α')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Number of peaks vs alpha
        ax = axes[2, col]
        ax.semilogx(r.alphas, r.n_peaks, 'o-', color=color)
        ax.axvline(r.alpha_lcurve, color='red', linestyle='--', alpha=0.7, label='L-curve')
        ax.axvline(r.alpha_discrepancy, color='green', linestyle=':', alpha=0.7, label='Discrepancy')
        ax.axhline(4, color='gray', linestyle='-', alpha=0.5, label='True (4)')
        ax.set_xlabel('α')
        ax.set_ylabel('Number of peaks')
        ax.set_title('Peak count vs α')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved parameter sweep plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_solutions_comparison(
    results: Dict[str, ParameterSweepResult],
    interior_points: np.ndarray,
    sources_true: List[Tuple] = None,
    param_type: str = 'lcurve',
    save_path: str = None,
    show: bool = True
):
    """
    Plot solutions at selected parameters.
    
    Parameters
    ----------
    results : dict
        {method: ParameterSweepResult}
    interior_points : array (n, 2)
        Source candidate positions
    sources_true : list, optional
        True sources for overlay
    param_type : str
        'lcurve', 'discrepancy', or 'best_rmse'
    save_path : str, optional
        Path to save figure
    show : bool
        Display the figure
    """
    import matplotlib.pyplot as plt
    
    try:
        from .comparison import find_intensity_peaks
    except ImportError:
        from comparison import find_intensity_peaks
    
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    titles = {'l1': 'L1 (Lasso)', 'l2': 'L2 (Tikhonov)', 'tv': 'TV'}
    
    for ax, method in zip(axes, methods):
        r = results[method]
        
        if param_type == 'lcurve':
            idx = r.idx_lcurve
            alpha = r.alpha_lcurve
        elif param_type == 'discrepancy':
            idx = r.idx_discrepancy
            alpha = r.alpha_discrepancy
        elif param_type == 'best_rmse':
            idx = r.idx_best_rmse
            alpha = r.alpha_best_rmse
        else:
            raise ValueError(f"Unknown param_type: {param_type}")
        
        q = r.solutions[idx]
        peaks = find_intensity_peaks(interior_points, q)
        
        # Plot intensity field
        vmax = max(np.max(np.abs(q)), 0.01)
        sc = ax.scatter(interior_points[:, 0], interior_points[:, 1], 
                       c=q, cmap='RdBu_r', s=30, vmin=-vmax, vmax=vmax)
        
        # True sources
        if sources_true is not None:
            for (x, y), intensity in sources_true:
                color = 'red' if intensity > 0 else 'blue'
                ax.plot(x, y, 'o', ms=15, mec=color, mfc='none', mew=2)
        
        # Detected peaks
        for peak in peaks:
            x, y = peak.position
            intensity = peak.integrated_intensity
            marker = '^' if intensity > 0 else 'v'
            color = 'darkred' if intensity > 0 else 'darkblue'
            ax.plot(x, y, marker, ms=8, color=color, alpha=0.7)
        
        # Unit circle (if applicable)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        rmse = r.position_rmses[idx] if not np.isnan(r.position_rmses[idx]) else 0
        title = titles.get(method, method.upper())
        ax.set_title(f'{title}\nα={alpha:.1e}, {len(peaks)} peaks, RMSE={rmse:.3f}')
        plt.colorbar(sc, ax=ax)
    
    plt.suptitle(f'Solutions at {param_type.replace("_", " ").title()} parameters', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved solutions comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def run_full_comparison(
    G: np.ndarray,
    u_measured: np.ndarray,
    interior_points: np.ndarray,
    noise_level: float = None,
    sources_true: List[Tuple] = None,
    alphas: np.ndarray = None,
    output_dir: str = '.',
    verbose: bool = True
) -> Dict[str, ParameterSweepResult]:
    """
    Run complete parameter sweep comparison for L1, L2, and TV.
    
    Generates:
    - L-curve plots for each method
    - RMSE vs alpha curves
    - Solution visualizations at optimal parameters
    
    Parameters
    ----------
    G : array (m, n)
        Green's function matrix
    u_measured : array (m,)
        Boundary measurements
    interior_points : array (n, 2)
        Source candidate positions
    noise_level : float, optional
        For discrepancy principle
    sources_true : list, optional
        For RMSE computation
    alphas : array, optional
        Regularization parameters to test
    output_dir : str
        Directory to save plots
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        {method: ParameterSweepResult}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Build gradient operator for TV
    D = build_gradient_operator(interior_points)
    
    results = {}
    
    for method in ['l1', 'l2', 'tv']:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running parameter sweep for {method.upper()}")
            print('='*60)
        
        results[method] = parameter_sweep(
            G, u_measured, interior_points,
            method=method,
            alphas=alphas,
            noise_level=noise_level,
            sources_true=sources_true,
            D=D if method == 'tv' else None,
            verbose=verbose
        )
    
    # Generate plots
    if verbose:
        print("\nGenerating diagnostic plots...")
    
    plot_parameter_sweep(
        results,
        save_path=os.path.join(output_dir, 'parameter_sweep.png'),
        show=False
    )
    
    for param_type in ['lcurve', 'discrepancy']:
        plot_solutions_comparison(
            results, interior_points, sources_true,
            param_type=param_type,
            save_path=os.path.join(output_dir, f'solutions_{param_type}.png'),
            show=False
        )
    
    if sources_true is not None:
        plot_solutions_comparison(
            results, interior_points, sources_true,
            param_type='best_rmse',
            save_path=os.path.join(output_dir, 'solutions_best_rmse.png'),
            show=False
        )
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY: Optimal Parameters")
        print("="*70)
        print(f"\n{'Method':<12} {'L-curve α':<12} {'Discrepancy α':<15} {'Best RMSE α':<12} {'Best RMSE':<10}")
        print("-"*70)
        for method in ['l1', 'l2', 'tv']:
            r = results[method]
            rmse = r.position_rmses[r.idx_best_rmse] if not np.isnan(r.position_rmses[r.idx_best_rmse]) else 0
            print(f"{method.upper():<12} {r.alpha_lcurve:<12.2e} {r.alpha_discrepancy:<15.2e} {r.alpha_best_rmse:<12.2e} {rmse:<10.4f}")
    
    return results
