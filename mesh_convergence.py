"""
Mesh Convergence Study for Inverse Source Localization
=======================================================

This module provides tools for:
1. FEM forward mesh convergence analysis
2. Source grid resolution analysis for the linear inverse problem
3. Optimal parameter selection based on convergence studies

The goal is to determine:
- Optimal FEM mesh resolution for accurate forward solutions
- Optimal source grid resolution for the distributed inverse problem
- Trade-offs between accuracy and computational cost

Author: Claude (Anthropic)
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from time import time
import os


@dataclass
class ConvergenceResult:
    """Results from a single convergence test."""
    resolution: float
    n_nodes: int
    n_elements: int
    n_interior_points: int
    n_boundary_points: int
    forward_error: float  # Error vs reference solution
    inverse_localization: float  # Localization score
    inverse_sparsity: float  # Sparsity ratio
    time_forward: float
    time_inverse: float
    boundary_values: np.ndarray
    source_positions: np.ndarray


@dataclass
class ConvergenceStudy:
    """Complete convergence study results."""
    domain_type: str
    resolutions: np.ndarray
    results: List[ConvergenceResult]
    reference_resolution: float
    optimal_forward_resolution: float
    optimal_source_resolution: float
    
    def plot(self, save_path: str = None, show: bool = True):
        """Plot convergence results."""
        plot_convergence_study(self, save_path, show)


def run_forward_mesh_convergence(domain_type: str = 'disk',
                                  domain_params: dict = None,
                                  sources: List[Tuple] = None,
                                  resolutions: np.ndarray = None,
                                  reference_resolution: float = 0.02,
                                  verbose: bool = True) -> ConvergenceStudy:
    """
    Study forward problem mesh convergence.
    
    Parameters
    ----------
    domain_type : str
        'disk', 'ellipse', 'square', 'polygon', 'star', 'brain'
    domain_params : dict
        Domain-specific parameters
    sources : list
        Test sources. If None, uses domain defaults.
    resolutions : array
        Mesh resolutions to test. Default: [0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04]
    reference_resolution : float
        Fine mesh for reference solution
    verbose : bool
        Print progress
        
    Returns
    -------
    study : ConvergenceStudy
        Complete convergence results
    """
    if resolutions is None:
        resolutions = np.array([0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04])
    
    if sources is None:
        try:
            from .comparison import create_domain_sources
        except ImportError:
            from comparison import create_domain_sources
        sources = create_domain_sources(domain_type, domain_params)
    
    if verbose:
        print("="*70)
        print(f"FORWARD MESH CONVERGENCE STUDY: {domain_type.upper()}")
        print("="*70)
        print(f"Sources: {len(sources)}")
        print(f"Resolutions: {resolutions}")
        print(f"Reference resolution: {reference_resolution}")
    
    # Compute reference solution
    if verbose:
        print(f"\nComputing reference solution (h={reference_resolution})...")
    
    ref_result = _solve_forward_at_resolution(
        domain_type, domain_params, sources, reference_resolution
    )
    u_ref = ref_result['boundary_values']
    
    if verbose:
        print(f"  Reference: {ref_result['n_nodes']} nodes, {ref_result['n_elements']} elements")
    
    # Test each resolution
    results = []
    for h in resolutions:
        if verbose:
            print(f"\nTesting resolution h={h}...")
        
        t0 = time()
        res = _solve_forward_at_resolution(domain_type, domain_params, sources, h)
        t_forward = time() - t0
        
        # Compute error vs reference
        # Interpolate to common boundary points for comparison
        from scipy.interpolate import interp1d
        n_compare = min(len(u_ref), len(res['boundary_values']))
        
        # Use boundary position for interpolation
        theta_ref = np.linspace(0, 2*np.pi, len(u_ref), endpoint=False)
        theta_test = np.linspace(0, 2*np.pi, len(res['boundary_values']), endpoint=False)
        
        interp = interp1d(theta_ref, u_ref, kind='linear', fill_value='extrapolate')
        u_ref_interp = interp(theta_test)
        
        error = np.linalg.norm(res['boundary_values'] - u_ref_interp) / np.linalg.norm(u_ref_interp)
        
        result = ConvergenceResult(
            resolution=h,
            n_nodes=res['n_nodes'],
            n_elements=res['n_elements'],
            n_interior_points=res.get('n_interior', 0),
            n_boundary_points=res.get('n_boundary', len(res['boundary_values'])),
            forward_error=error,
            inverse_localization=0.0,  # Will be filled by inverse study
            inverse_sparsity=0.0,
            time_forward=t_forward,
            time_inverse=0.0,
            boundary_values=res['boundary_values'],
            source_positions=np.array([])
        )
        results.append(result)
        
        if verbose:
            print(f"  Nodes: {res['n_nodes']}, Elements: {res['n_elements']}")
            print(f"  Relative error: {error:.2e}")
            print(f"  Time: {t_forward:.2f}s")
    
    # Determine optimal resolution (error < 1% with minimum nodes)
    errors = np.array([r.forward_error for r in results])
    
    # Find finest resolution where error drops below 1%
    optimal_idx = len(results) - 1
    for i, err in enumerate(errors):
        if err < 0.01:
            optimal_idx = i
            break
    
    optimal_resolution = results[optimal_idx].resolution
    
    if verbose:
        print("\n" + "="*70)
        print("FORWARD CONVERGENCE SUMMARY")
        print("="*70)
        print(f"\n{'Resolution':<12} {'Nodes':<10} {'Elements':<10} {'Error':<12} {'Time (s)':<10}")
        print("-"*54)
        for r in results:
            print(f"{r.resolution:<12.3f} {r.n_nodes:<10} {r.n_elements:<10} {r.forward_error:<12.2e} {r.time_forward:<10.2f}")
        print(f"\nOptimal forward resolution: {optimal_resolution} (error < 1%)")
    
    return ConvergenceStudy(
        domain_type=domain_type,
        resolutions=resolutions,
        results=results,
        reference_resolution=reference_resolution,
        optimal_forward_resolution=optimal_resolution,
        optimal_source_resolution=0.15  # Default, will be updated by inverse study
    )


def run_inverse_source_grid_convergence(domain_type: str = 'disk',
                                         domain_params: dict = None,
                                         sources_true: List[Tuple] = None,
                                         source_resolutions: np.ndarray = None,
                                         forward_resolution: float = 0.08,
                                         noise_level: float = 0.001,
                                         alpha: float = 1e-4,
                                         method: str = 'l1',
                                         seed: int = 42,
                                         verbose: bool = True) -> ConvergenceStudy:
    """
    Study source grid resolution convergence for the distributed inverse problem.
    
    Parameters
    ----------
    domain_type : str
        Domain type
    domain_params : dict
        Domain parameters
    sources_true : list
        True source configuration
    source_resolutions : array
        Source grid resolutions to test
    forward_resolution : float
        FEM mesh resolution for forward problem
    noise_level : float
        Measurement noise level
    alpha : float
        Regularization parameter
    method : str
        'l1', 'l2', or 'tv'
    seed : int
        Random seed
    verbose : bool
        Print progress
        
    Returns
    -------
    study : ConvergenceStudy
        Convergence results
    """
    if source_resolutions is None:
        source_resolutions = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08])
    
    if sources_true is None:
        try:
            from .comparison import create_domain_sources
        except ImportError:
            from comparison import create_domain_sources
        sources_true = create_domain_sources(domain_type, domain_params)
    
    if verbose:
        print("="*70)
        print(f"SOURCE GRID CONVERGENCE STUDY: {domain_type.upper()}")
        print("="*70)
        print(f"True sources: {len(sources_true)}")
        print(f"Forward resolution: {forward_resolution}")
        print(f"Source resolutions: {source_resolutions}")
        print(f"Method: {method.upper()}, α={alpha:.1e}")
        print(f"Noise level: {noise_level}")
    
    # Generate synthetic data
    np.random.seed(seed)
    u_clean = _solve_forward_at_resolution(
        domain_type, domain_params, sources_true, forward_resolution
    )['boundary_values']
    u_measured = u_clean + noise_level * np.random.randn(len(u_clean))
    
    results = []
    for h_src in source_resolutions:
        if verbose:
            print(f"\nTesting source resolution h={h_src}...")
        
        t0 = time()
        inv_result = _solve_inverse_at_resolution(
            domain_type, domain_params, u_measured, sources_true,
            forward_resolution, h_src, alpha, method
        )
        t_inverse = time() - t0
        
        result = ConvergenceResult(
            resolution=h_src,
            n_nodes=inv_result['n_nodes'],
            n_elements=inv_result['n_elements'],
            n_interior_points=inv_result['n_source_points'],
            n_boundary_points=inv_result['n_boundary'],
            forward_error=0.0,
            inverse_localization=inv_result['localization'],
            inverse_sparsity=inv_result['sparsity'],
            time_forward=0.0,
            time_inverse=t_inverse,
            boundary_values=np.array([]),
            source_positions=inv_result['interior_points']
        )
        results.append(result)
        
        if verbose:
            print(f"  Source points: {inv_result['n_source_points']}")
            print(f"  Localization: {inv_result['localization']:.4f}")
            print(f"  Sparsity: {inv_result['sparsity']:.4f}")
            print(f"  Time: {t_inverse:.2f}s")
    
    # Find optimal: highest localization with reasonable time
    localizations = np.array([r.inverse_localization for r in results])
    
    # Optimal = resolution with highest localization that doesn't take >10x the coarsest
    time_threshold = results[0].time_inverse * 10
    optimal_idx = 0
    best_loc = 0
    for i, r in enumerate(results):
        if r.time_inverse < time_threshold and r.inverse_localization > best_loc:
            best_loc = r.inverse_localization
            optimal_idx = i
    
    optimal_resolution = results[optimal_idx].resolution
    
    if verbose:
        print("\n" + "="*70)
        print("SOURCE GRID CONVERGENCE SUMMARY")
        print("="*70)
        print(f"\n{'Resolution':<12} {'Points':<10} {'Localization':<14} {'Sparsity':<12} {'Time (s)':<10}")
        print("-"*58)
        for r in results:
            print(f"{r.resolution:<12.3f} {r.n_interior_points:<10} {r.inverse_localization:<14.4f} {r.inverse_sparsity:<12.4f} {r.time_inverse:<10.2f}")
        print(f"\nOptimal source resolution: {optimal_resolution}")
    
    return ConvergenceStudy(
        domain_type=domain_type,
        resolutions=source_resolutions,
        results=results,
        reference_resolution=source_resolutions[-1],
        optimal_forward_resolution=forward_resolution,
        optimal_source_resolution=optimal_resolution
    )


def _solve_forward_at_resolution(domain_type: str, domain_params: dict,
                                  sources: List[Tuple], resolution: float) -> dict:
    """Solve forward problem at given resolution."""
    if domain_type == 'disk':
        try:
            from .fem_solver import FEMForwardSolver
        except ImportError:
            from fem_solver import FEMForwardSolver
        
        solver = FEMForwardSolver(resolution=resolution, verbose=False)
        u = solver.solve(sources)
        
        return {
            'boundary_values': u,
            'n_nodes': len(solver.nodes) if hasattr(solver, 'nodes') else 0,
            'n_elements': len(solver.elements) if hasattr(solver, 'elements') else 0,
            'n_interior': len(solver.interior_indices) if hasattr(solver, 'interior_indices') else 0,
            'n_boundary': len(solver.boundary_indices) if hasattr(solver, 'boundary_indices') else 0
        }
    
    elif domain_type == 'ellipse':
        try:
            from .fem_solver import FEMForwardSolver
            from .mesh import create_ellipse_mesh
        except ImportError:
            from fem_solver import FEMForwardSolver
            from mesh import create_ellipse_mesh
        
        a = domain_params.get('a', 2.0) if domain_params else 2.0
        b = domain_params.get('b', 1.0) if domain_params else 1.0
        
        mesh_data = create_ellipse_mesh(a, b, resolution)
        solver = FEMForwardSolver(resolution=resolution, verbose=False, mesh_data=mesh_data)
        u = solver.solve(sources)
        
        return {
            'boundary_values': u,
            'n_nodes': len(mesh_data[0]),
            'n_elements': len(mesh_data[1]),
            'n_interior': len(mesh_data[3]),
            'n_boundary': len(mesh_data[2])
        }
    
    elif domain_type in ['square', 'polygon', 'star', 'brain']:
        try:
            from .fem_solver import FEMForwardSolver
            from .mesh import create_polygon_mesh, get_brain_boundary
        except ImportError:
            from fem_solver import FEMForwardSolver
            from mesh import create_polygon_mesh, get_brain_boundary
        
        if domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        elif domain_type == 'polygon':
            vertices = domain_params.get('vertices') if domain_params else [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        elif domain_type == 'star':
            n_petals = domain_params.get('n_petals', 5) if domain_params else 5
            amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
            n_v = 100
            theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
            r_v = 1.0 + amplitude * np.cos(n_petals * theta_v)
            vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
        else:  # brain
            boundary = get_brain_boundary(n_points=100)
            vertices = [tuple(p) for p in boundary]
        
        mesh_data = create_polygon_mesh(vertices, resolution)
        solver = FEMForwardSolver(resolution=resolution, verbose=False, mesh_data=mesh_data)
        u = solver.solve(sources)
        
        return {
            'boundary_values': u,
            'n_nodes': len(mesh_data[0]),
            'n_elements': len(mesh_data[1]),
            'n_interior': len(mesh_data[3]) if len(mesh_data) > 3 else 0,
            'n_boundary': len(mesh_data[2])
        }
    
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")


def _solve_inverse_at_resolution(domain_type: str, domain_params: dict,
                                  u_measured: np.ndarray, sources_true: List[Tuple],
                                  forward_resolution: float, source_resolution: float,
                                  alpha: float, method: str) -> dict:
    """Solve inverse problem at given source grid resolution."""
    try:
        from .fem_solver import FEMLinearInverseSolver
        from .parameter_selection import localization_score, sparsity_ratio
        from .mesh import get_brain_boundary
    except ImportError:
        from fem_solver import FEMLinearInverseSolver
        from parameter_selection import localization_score, sparsity_ratio
        from mesh import get_brain_boundary
    
    # Create solver with specified resolutions
    if domain_type == 'disk':
        solver = FEMLinearInverseSolver(forward_resolution=forward_resolution,
                                         source_resolution=source_resolution,
                                         verbose=False)
    elif domain_type == 'ellipse':
        a = domain_params.get('a', 2.0) if domain_params else 2.0
        b = domain_params.get('b', 1.0) if domain_params else 1.0
        solver = FEMLinearInverseSolver.from_ellipse(a, b,
                                                      forward_resolution=forward_resolution,
                                                      source_resolution=source_resolution,
                                                      verbose=False)
    elif domain_type in ['square', 'polygon', 'star', 'brain']:
        if domain_type == 'square':
            vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        elif domain_type == 'polygon':
            vertices = domain_params.get('vertices') if domain_params else [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        elif domain_type == 'star':
            n_petals = domain_params.get('n_petals', 5) if domain_params else 5
            amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
            n_v = 100
            theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
            r_v = 1.0 + amplitude * np.cos(n_petals * theta_v)
            vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
        else:  # brain
            boundary = get_brain_boundary(n_points=100)
            vertices = [tuple(p) for p in boundary]
        
        solver = FEMLinearInverseSolver.from_polygon(vertices,
                                                      forward_resolution=forward_resolution,
                                                      source_resolution=source_resolution,
                                                      verbose=False)
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")
    
    solver.build_greens_matrix(verbose=False)
    
    # Solve
    if method == 'l1':
        q = solver.solve_l1(u_measured, alpha=alpha)
    elif method == 'l2':
        q = solver.solve_l2(u_measured, alpha=alpha)
    elif method == 'tv':
        q = solver.solve_tv(u_measured, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute metrics
    loc = localization_score(q, solver.interior_points, sources_true)
    spar = sparsity_ratio(q, target_sources=len(sources_true))
    
    return {
        'n_nodes': len(solver.nodes) if hasattr(solver, 'nodes') else 0,
        'n_elements': len(solver.elements) if hasattr(solver, 'elements') else 0,
        'n_source_points': len(solver.interior_points),
        'n_boundary': len(solver.boundary_indices) if hasattr(solver, 'boundary_indices') else 0,
        'localization': loc,
        'sparsity': spar,
        'interior_points': solver.interior_points.copy(),
        'q': q.copy()
    }


def plot_convergence_study(study: ConvergenceStudy, save_path: str = None, show: bool = True):
    """
    Plot convergence study results.
    
    Parameters
    ----------
    study : ConvergenceStudy
        Results to plot
    save_path : str, optional
        Path to save figure
    show : bool
        Display figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    resolutions = [r.resolution for r in study.results]
    
    # Check if this is a forward or inverse study
    is_forward = study.results[0].forward_error > 0
    is_inverse = study.results[0].inverse_localization > 0
    
    if is_forward:
        # Plot 1: Forward error vs resolution
        ax = axes[0, 0]
        errors = [r.forward_error for r in study.results]
        ax.loglog(resolutions, errors, 'bo-', linewidth=2, markersize=8)
        ax.axhline(0.01, color='r', linestyle='--', label='1% error threshold')
        ax.axvline(study.optimal_forward_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_forward_resolution}')
        ax.set_xlabel('Mesh resolution h', fontsize=12)
        ax.set_ylabel('Relative error', fontsize=12)
        ax.set_title('Forward Solution Convergence', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 2: Nodes vs resolution
        ax = axes[0, 1]
        nodes = [r.n_nodes for r in study.results]
        ax.semilogy(resolutions, nodes, 'gs-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_forward_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_forward_resolution}')
        ax.set_xlabel('Mesh resolution h', fontsize=12)
        ax.set_ylabel('Number of nodes', fontsize=12)
        ax.set_title('Mesh Size vs Resolution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 3: Time vs resolution
        ax = axes[1, 0]
        times = [r.time_forward for r in study.results]
        ax.semilogy(resolutions, times, 'r^-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_forward_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_forward_resolution}')
        ax.set_xlabel('Mesh resolution h', fontsize=12)
        ax.set_ylabel('Computation time (s)', fontsize=12)
        ax.set_title('Forward Solve Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 4: Error vs nodes (efficiency)
        ax = axes[1, 1]
        ax.loglog(nodes, errors, 'mo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of nodes', fontsize=12)
        ax.set_ylabel('Relative error', fontsize=12)
        ax.set_title('Efficiency: Error vs Computational Cost', fontsize=14)
        ax.axhline(0.01, color='r', linestyle='--', label='1% error')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if is_inverse:
        # Plot 1: Localization vs resolution
        ax = axes[0, 0]
        locs = [r.inverse_localization for r in study.results]
        ax.plot(resolutions, locs, 'bo-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_source_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_source_resolution}')
        ax.set_xlabel('Source grid resolution h', fontsize=12)
        ax.set_ylabel('Localization score', fontsize=12)
        ax.set_title('Inverse Solution Quality (Localization)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 2: Sparsity vs resolution
        ax = axes[0, 1]
        spars = [r.inverse_sparsity for r in study.results]
        ax.plot(resolutions, spars, 'gs-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_source_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_source_resolution}')
        ax.set_xlabel('Source grid resolution h', fontsize=12)
        ax.set_ylabel('Sparsity ratio', fontsize=12)
        ax.set_title('Solution Sparsity', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 3: Time vs resolution
        ax = axes[1, 0]
        times = [r.time_inverse for r in study.results]
        ax.semilogy(resolutions, times, 'r^-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_source_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_source_resolution}')
        ax.set_xlabel('Source grid resolution h', fontsize=12)
        ax.set_ylabel('Computation time (s)', fontsize=12)
        ax.set_title('Inverse Solve Time', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
        
        # Plot 4: Source points vs resolution
        ax = axes[1, 1]
        n_pts = [r.n_interior_points for r in study.results]
        ax.semilogy(resolutions, n_pts, 'mo-', linewidth=2, markersize=8)
        ax.axvline(study.optimal_source_resolution, color='g', linestyle=':', 
                   label=f'Optimal h={study.optimal_source_resolution}')
        ax.set_xlabel('Source grid resolution h', fontsize=12)
        ax.set_ylabel('Number of source points', fontsize=12)
        ax.set_title('Source Grid Size', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()
    
    plt.suptitle(f'{study.domain_type.upper()} Domain Convergence Study', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def run_full_convergence_study(domain_type: str = 'disk',
                                domain_params: dict = None,
                                output_dir: str = 'results/convergence',
                                verbose: bool = True) -> Tuple[ConvergenceStudy, ConvergenceStudy]:
    """
    Run complete forward and inverse convergence studies.
    
    Parameters
    ----------
    domain_type : str
        Domain to study
    domain_params : dict
        Domain parameters
    output_dir : str
        Directory for output files
    verbose : bool
        Print progress
        
    Returns
    -------
    forward_study, inverse_study : tuple of ConvergenceStudy
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Forward convergence
    forward_study = run_forward_mesh_convergence(
        domain_type=domain_type,
        domain_params=domain_params,
        verbose=verbose
    )
    forward_study.plot(
        save_path=os.path.join(output_dir, f'{domain_type}_forward_convergence.png'),
        show=False
    )
    
    # Inverse convergence (using optimal forward resolution)
    inverse_study = run_inverse_source_grid_convergence(
        domain_type=domain_type,
        domain_params=domain_params,
        forward_resolution=forward_study.optimal_forward_resolution,
        verbose=verbose
    )
    inverse_study.plot(
        save_path=os.path.join(output_dir, f'{domain_type}_inverse_convergence.png'),
        show=False
    )
    
    # Summary
    if verbose:
        print("\n" + "="*70)
        print("CONVERGENCE STUDY COMPLETE")
        print("="*70)
        print(f"\nDomain: {domain_type}")
        print(f"Optimal forward mesh resolution: {forward_study.optimal_forward_resolution}")
        print(f"Optimal source grid resolution: {inverse_study.optimal_source_resolution}")
        print(f"\nFigures saved to: {output_dir}/")
    
    return forward_study, inverse_study


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesh convergence study')
    parser.add_argument('--domain', type=str, default='disk',
                       choices=['disk', 'ellipse', 'star', 'square', 'polygon', 'brain'],
                       help='Domain type')
    parser.add_argument('--output-dir', type=str, default='results/convergence',
                       help='Output directory')
    parser.add_argument('--forward-only', action='store_true',
                       help='Run only forward convergence')
    parser.add_argument('--inverse-only', action='store_true',
                       help='Run only inverse convergence')
    
    args = parser.parse_args()
    
    if args.forward_only:
        study = run_forward_mesh_convergence(domain_type=args.domain)
        study.plot(save_path=os.path.join(args.output_dir, f'{args.domain}_forward_convergence.png'))
    elif args.inverse_only:
        study = run_inverse_source_grid_convergence(domain_type=args.domain)
        study.plot(save_path=os.path.join(args.output_dir, f'{args.domain}_inverse_convergence.png'))
    else:
        run_full_convergence_study(domain_type=args.domain, output_dir=args.output_dir)
