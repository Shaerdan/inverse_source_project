#!/usr/bin/env python3
"""
Comprehensive Inverse Source Localization Comparison
=====================================================

This script generates comparison plots and L-curve analyses for different domains.
Uses the existing package infrastructure with proper gmsh mesh generation.

Usage:
    python generate_comparison.py --shape disk
    python generate_comparison.py --shape ellipse
    python generate_comparison.py --shape brain
    python generate_comparison.py --shape all

Output:
    Results saved to: ./results_{shape}_{hash}/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
import hashlib
from datetime import datetime
from time import time
import warnings
warnings.filterwarnings('ignore')

# Import from the package
from comparison import (
    compare_all_solvers_general, 
    create_domain_sources,
    get_sensor_locations,
    ComparisonResult,
    plot_comparison
)
from mesh import get_brain_boundary, create_disk_mesh, create_ellipse_mesh, create_brain_mesh
from fem_solver import FEMLinearInverseSolver, FEMNonlinearInverseSolver, FEMForwardSolver
from analytical_solver import (AnalyticalForwardSolver, AnalyticalLinearInverseSolver, 
                                AnalyticalNonlinearInverseSolver)
from conformal_solver import (EllipseMap, MFSConformalMap, ConformalForwardSolver,
                              ConformalLinearInverseSolver, ConformalNonlinearInverseSolver)
from parameter_selection import find_lcurve_corner, build_gradient_operator


def generate_run_hash():
    """Generate a short hash for this run."""
    timestamp = datetime.now().isoformat()
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]


def create_output_dir(shape):
    """Create output directory with shape name and hash."""
    run_hash = generate_run_hash()
    dir_name = f"./results_{shape}_{run_hash}"
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def compute_l_curve_proper(solver, u_measured, method='l2', alphas=None, verbose=True):
    """
    Compute L-curve using proper CVXPY optimization with zero-sum constraint.
    """
    import cvxpy as cp
    
    if alphas is None:
        alphas = np.logspace(-6, -1, 20)
    
    G = solver.G
    interior_points = solver.interior_points
    u_centered = u_measured - np.mean(u_measured)
    
    # Build gradient operator for TV
    D = build_gradient_operator(interior_points)
    
    residuals = []
    regularizers = []
    solutions = []
    
    for alpha in alphas:
        n = G.shape[1]
        q_var = cp.Variable(n)
        constraints = [cp.sum(q_var) == 0]
        
        if method == 'l1':
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                alpha * cp.norm1(q_var)
            )
        elif method == 'l2':
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                0.5 * alpha * cp.sum_squares(q_var)
            )
        else:  # tv
            objective = cp.Minimize(
                0.5 * cp.sum_squares(G @ q_var - u_centered) + 
                alpha * cp.norm1(D @ q_var)
            )
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if q_var.value is not None:
                q = q_var.value
                res = np.linalg.norm(G @ q - u_centered)
                if method == 'l1':
                    reg = np.sum(np.abs(q))
                elif method == 'l2':
                    reg = np.linalg.norm(q)
                else:
                    reg = np.sum(np.abs(D @ q))
                residuals.append(res)
                regularizers.append(reg)
                solutions.append(q)
            else:
                residuals.append(np.nan)
                regularizers.append(np.nan)
                solutions.append(None)
        except:
            residuals.append(np.nan)
            regularizers.append(np.nan)
            solutions.append(None)
    
    residuals = np.array(residuals)
    regularizers = np.array(regularizers)
    
    # Find L-curve corner
    # NOTE: find_lcurve_corner already does log10 internally, so pass RAW values
    valid = ~np.isnan(residuals) & ~np.isnan(regularizers)
    if np.sum(valid) > 3:
        corner_idx = find_lcurve_corner(
            residuals[valid],  # RAW values - function does log internally
            regularizers[valid]
        )
        # Map back to original index
        valid_indices = np.where(valid)[0]
        optimal_idx = valid_indices[corner_idx] if corner_idx < len(valid_indices) else len(alphas) // 2
    else:
        optimal_idx = len(alphas) // 2
    
    return alphas, residuals, regularizers, optimal_idx, solutions


def plot_l_curve(alphas, residuals, regularizers, optimal_idx, method, domain, save_path):
    """Plot L-curve with optimal alpha marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    valid = ~np.isnan(residuals) & ~np.isnan(regularizers)
    
    ax.loglog(residuals[valid], regularizers[valid], 'b.-', linewidth=1.5, markersize=6)
    if optimal_idx < len(residuals) and valid[optimal_idx]:
        ax.loglog(residuals[optimal_idx], regularizers[optimal_idx], 'ro', 
                  markersize=12, label=f'Optimal α = {alphas[optimal_idx]:.2e}')
    
    # Annotate some alpha values
    valid_indices = np.where(valid)[0]
    for i in [0, len(valid_indices)//4, len(valid_indices)//2, 3*len(valid_indices)//4, -1]:
        if i < len(valid_indices):
            idx = valid_indices[i]
            ax.annotate(f'α={alphas[idx]:.1e}', 
                       (residuals[idx], regularizers[idx]),
                       textcoords='offset points', xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('Residual Norm ||Gq - u||', fontsize=12)
    ax.set_ylabel('Regularizer Norm', fontsize=12)
    ax.set_title(f'L-Curve for {method.upper()} Regularization ({domain})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    return alphas[optimal_idx] if optimal_idx < len(alphas) else alphas[len(alphas)//2]


def run_disk_comparison(output_dir, n_sources=4, noise_level=0.01, seed=42, 
                        forward_resolution=0.05, source_resolution=0.1):
    """Run full comparison for disk domain using proper gmsh meshes."""
    print("\n" + "="*70)
    print("DISK DOMAIN COMPARISON")
    print("="*70)
    
    # Create sources
    sources = create_domain_sources('disk', n_sources=n_sources, seed=seed)
    print(f"True sources ({n_sources}):")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    n_sensors = 200
    
    results = {}
    l_curve_data = {}
    
    # =========================================================================
    # ANALYTICAL METHODS (exact Green's function)
    # =========================================================================
    print("\n--- Analytical Methods ---")
    
    # Forward solver
    analytical_forward = AnalyticalForwardSolver(n_sensors)
    u_analytical = analytical_forward.solve(sources)
    np.random.seed(seed)
    u_analytical_noisy = u_analytical + noise_level * np.std(u_analytical) * np.random.randn(len(u_analytical))
    
    # Linear solver (uses gmsh for source grid)
    analytical_linear = AnalyticalLinearInverseSolver(
        n_boundary=n_sensors, 
        source_resolution=source_resolution,
        verbose=True
    )
    analytical_linear.build_greens_matrix(verbose=True)
    
    print(f"  Source grid points: {len(analytical_linear.interior_points)}")
    
    # L-curve analysis
    print("\n  L-curve analysis for Analytical...")
    alphas = np.logspace(-6, -1, 20)
    
    for method in ['l2', 'l1', 'tv']:
        print(f"    Computing L-curve for {method.upper()}...")
        alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
            analytical_linear, u_analytical_noisy, method=method, alphas=alphas
        )
        
        save_path = os.path.join(output_dir, f'l_curve_disk_analytical_{method}.png')
        opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                 method, 'Disk (Analytical)', save_path)
        l_curve_data[f'analytical_{method}'] = {
            'alphas': alphas_out,
            'residuals': residuals,
            'regularizers': regularizers,
            'optimal_idx': opt_idx,
            'optimal_alpha': opt_alpha
        }
        
        # Store solution at optimal alpha
        if solutions[opt_idx] is not None:
            results[f'Analytical_{method.upper()}'] = {
                'grid_intensities': solutions[opt_idx],
                'grid_positions': analytical_linear.interior_points,
                'alpha': opt_alpha,
                'time': 0.0
            }
    
    # Analytical nonlinear
    print("\n  Running Analytical Nonlinear...")
    analytical_nonlinear = AnalyticalNonlinearInverseSolver(n_sources, n_sensors)
    analytical_nonlinear.set_measured_data(u_analytical_noisy)
    
    t0 = time()
    result_nl = analytical_nonlinear.solve(method='SLSQP', n_restarts=10, maxiter=10000)
    elapsed = time() - t0
    
    from scipy.optimize import linear_sum_assignment
    rmse = compute_position_rmse(sources, result_nl.sources)
    results['Analytical_Nonlinear'] = {
        'sources': result_nl.sources,
        'rmse': rmse,
        'time': elapsed
    }
    print(f"    RMSE: {rmse:.2e}, Time: {elapsed:.2f}s")
    
    # =========================================================================
    # FEM METHODS (gmsh mesh)
    # =========================================================================
    print("\n--- FEM Methods ---")
    
    # FEM Linear solver (uses gmsh for both forward and source mesh)
    fem_linear = FEMLinearInverseSolver(
        forward_resolution=forward_resolution,
        source_resolution=source_resolution,
        verbose=True
    )
    fem_linear.build_greens_matrix(verbose=True)
    
    print(f"  FEM mesh nodes: {len(fem_linear.nodes)}")
    print(f"  FEM source grid points: {len(fem_linear.interior_points)}")
    
    # Generate measurements using FEM forward
    mesh_data = (fem_linear.nodes, fem_linear.elements, fem_linear.boundary_indices,
                 np.setdiff1d(np.arange(len(fem_linear.nodes)), fem_linear.boundary_indices))
    fem_forward = FEMForwardSolver(resolution=forward_resolution, verbose=False, mesh_data=mesh_data)
    u_fem = fem_forward.solve(sources)
    np.random.seed(seed)
    u_fem_noisy = u_fem + noise_level * np.std(u_fem) * np.random.randn(len(u_fem))
    
    # L-curve for FEM
    print("\n  L-curve analysis for FEM...")
    for method in ['l2', 'l1', 'tv']:
        print(f"    Computing L-curve for {method.upper()}...")
        alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
            fem_linear, u_fem_noisy, method=method, alphas=alphas
        )
        
        save_path = os.path.join(output_dir, f'l_curve_disk_fem_{method}.png')
        opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                 method, 'Disk (FEM)', save_path)
        l_curve_data[f'fem_{method}'] = {
            'optimal_alpha': opt_alpha
        }
        
        if solutions[opt_idx] is not None:
            results[f'FEM_{method.upper()}'] = {
                'grid_intensities': solutions[opt_idx],
                'grid_positions': fem_linear.interior_points,
                'alpha': opt_alpha,
                'time': 0.0
            }
    
    # FEM Nonlinear
    print("\n  Running FEM Nonlinear...")
    fem_nonlinear = FEMNonlinearInverseSolver(
        n_sources, 
        resolution=forward_resolution,
        n_sensors=n_sensors,
        verbose=False
    )
    u_fem_nl = fem_nonlinear.forward.solve_at_sensors(sources)
    np.random.seed(seed)
    u_fem_nl_noisy = u_fem_nl + noise_level * np.std(u_fem_nl) * np.random.randn(len(u_fem_nl))
    fem_nonlinear.set_measured_data(u_fem_nl_noisy)
    
    t0 = time()
    result_fem_nl = fem_nonlinear.solve(method='SLSQP', n_restarts=10, maxiter=5000)
    elapsed = time() - t0
    
    rmse_fem = compute_position_rmse(sources, result_fem_nl.sources)
    results['FEM_Nonlinear'] = {
        'sources': result_fem_nl.sources,
        'rmse': rmse_fem,
        'time': elapsed
    }
    print(f"    RMSE: {rmse_fem:.2e}, Time: {elapsed:.2f}s")
    
    # Create comparison plot
    create_comparison_plot_custom(sources, results, 'disk', output_dir)
    
    # Save summary
    save_summary(results, l_curve_data, 'disk', output_dir)
    
    return results, l_curve_data


def run_ellipse_comparison(output_dir, n_sources=4, noise_level=0.01, seed=42,
                           forward_resolution=0.05, source_resolution=0.1):
    """Run full comparison for ellipse domain."""
    print("\n" + "="*70)
    print("ELLIPSE DOMAIN COMPARISON")
    print("="*70)
    
    a, b = 1.5, 0.8
    domain_params = {'a': a, 'b': b}
    
    # Create sources
    sources = create_domain_sources('ellipse', domain_params, n_sources=n_sources, seed=seed)
    print(f"True sources ({n_sources}):")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    n_sensors = 200
    
    results = {}
    l_curve_data = {}
    alphas = np.logspace(-6, -1, 20)
    
    # =========================================================================
    # CONFORMAL METHODS
    # =========================================================================
    print("\n--- Conformal Methods ---")
    
    emap = EllipseMap(a=a, b=b)
    
    # Forward solver
    conformal_forward = ConformalForwardSolver(emap, n_sensors)
    u_conformal = conformal_forward.solve(sources)
    np.random.seed(seed)
    u_conformal_noisy = u_conformal + noise_level * np.std(u_conformal) * np.random.randn(len(u_conformal))
    
    # Linear solver
    conformal_linear = ConformalLinearInverseSolver(
        emap, n_boundary=n_sensors,
        source_resolution=source_resolution,
        verbose=True
    )
    conformal_linear.build_greens_matrix()
    
    print(f"  Source grid points: {len(conformal_linear.interior_points)}")
    
    # L-curve analysis
    print("\n  L-curve analysis for Conformal...")
    for method in ['l2', 'l1', 'tv']:
        print(f"    Computing L-curve for {method.upper()}...")
        alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
            conformal_linear, u_conformal_noisy, method=method, alphas=alphas
        )
        
        save_path = os.path.join(output_dir, f'l_curve_ellipse_conformal_{method}.png')
        opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                 method, 'Ellipse (Conformal)', save_path)
        l_curve_data[f'conformal_{method}'] = {'optimal_alpha': opt_alpha}
        
        if solutions[opt_idx] is not None:
            results[f'Conformal_{method.upper()}'] = {
                'grid_intensities': solutions[opt_idx],
                'grid_positions': conformal_linear.interior_points,
                'alpha': opt_alpha,
                'time': 0.0
            }
    
    # Conformal Nonlinear
    print("\n  Running Conformal Nonlinear...")
    conformal_nonlinear = ConformalNonlinearInverseSolver(emap, n_sources, n_sensors)
    
    t0 = time()
    sources_rec, residual = conformal_nonlinear.solve(u_conformal_noisy, method='SLSQP', n_restarts=10)
    elapsed = time() - t0
    
    rmse = compute_position_rmse(sources, sources_rec)
    results['Conformal_Nonlinear'] = {
        'sources': sources_rec,
        'rmse': rmse,
        'time': elapsed
    }
    print(f"    RMSE: {rmse:.2e}, Time: {elapsed:.2f}s")
    
    # =========================================================================
    # FEM METHODS
    # =========================================================================
    print("\n--- FEM Methods ---")
    
    # FEM Linear (uses gmsh for ellipse mesh)
    fem_linear = FEMLinearInverseSolver.from_ellipse(
        a, b,
        forward_resolution=forward_resolution,
        source_resolution=source_resolution,
        verbose=True
    )
    fem_linear.build_greens_matrix(verbose=True)
    
    print(f"  FEM mesh nodes: {len(fem_linear.nodes)}")
    print(f"  FEM source grid points: {len(fem_linear.interior_points)}")
    
    # Generate measurements
    mesh_data = (fem_linear.nodes, fem_linear.elements, fem_linear.boundary_indices,
                 np.setdiff1d(np.arange(len(fem_linear.nodes)), fem_linear.boundary_indices))
    fem_forward = FEMForwardSolver(resolution=forward_resolution, verbose=False, mesh_data=mesh_data)
    u_fem = fem_forward.solve(sources)
    np.random.seed(seed)
    u_fem_noisy = u_fem + noise_level * np.std(u_fem) * np.random.randn(len(u_fem))
    
    # L-curve for FEM
    print("\n  L-curve analysis for FEM...")
    for method in ['l2', 'l1', 'tv']:
        print(f"    Computing L-curve for {method.upper()}...")
        alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
            fem_linear, u_fem_noisy, method=method, alphas=alphas
        )
        
        save_path = os.path.join(output_dir, f'l_curve_ellipse_fem_{method}.png')
        opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                 method, 'Ellipse (FEM)', save_path)
        l_curve_data[f'fem_{method}'] = {'optimal_alpha': opt_alpha}
        
        if solutions[opt_idx] is not None:
            results[f'FEM_{method.upper()}'] = {
                'grid_intensities': solutions[opt_idx],
                'grid_positions': fem_linear.interior_points,
                'alpha': opt_alpha,
                'time': 0.0
            }
    
    # FEM Nonlinear
    print("\n  Running FEM Nonlinear...")
    fem_nonlinear = FEMNonlinearInverseSolver.from_ellipse(
        a, b, n_sources,
        resolution=forward_resolution,
        n_sensors=n_sensors,
        verbose=False
    )
    u_fem_nl = fem_nonlinear.forward.solve_at_sensors(sources)
    np.random.seed(seed)
    u_fem_nl_noisy = u_fem_nl + noise_level * np.std(u_fem_nl) * np.random.randn(len(u_fem_nl))
    fem_nonlinear.set_measured_data(u_fem_nl_noisy)
    
    t0 = time()
    result_fem_nl = fem_nonlinear.solve(method='SLSQP', n_restarts=10, maxiter=5000)
    elapsed = time() - t0
    
    rmse_fem = compute_position_rmse(sources, result_fem_nl.sources)
    results['FEM_Nonlinear'] = {
        'sources': result_fem_nl.sources,
        'rmse': rmse_fem,
        'time': elapsed
    }
    print(f"    RMSE: {rmse_fem:.2e}, Time: {elapsed:.2f}s")
    
    # Create comparison plot
    create_comparison_plot_custom(sources, results, 'ellipse', output_dir, 
                                  domain_params={'a': a, 'b': b})
    
    save_summary(results, l_curve_data, 'ellipse', output_dir)
    
    return results, l_curve_data


def run_brain_comparison(output_dir, n_sources=4, noise_level=0.01, seed=42,
                         forward_resolution=0.05, source_resolution=0.1):
    """Run full comparison for brain domain."""
    print("\n" + "="*70)
    print("BRAIN DOMAIN COMPARISON")
    print("="*70)
    
    # Get brain boundary
    brain_boundary = get_brain_boundary(n_points=100)
    vertices = [tuple(p) for p in brain_boundary]
    
    # Create sources inside brain
    sources = create_domain_sources('brain', n_sources=n_sources, seed=seed)
    print(f"True sources ({n_sources}):")
    for i, ((x, y), q) in enumerate(sources):
        print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
    
    n_sensors = 200
    
    results = {}
    l_curve_data = {}
    alphas = np.logspace(-6, -1, 20)
    
    # =========================================================================
    # CONFORMAL METHODS (MFS-based map)
    # =========================================================================
    print("\n--- Conformal Methods (MFS) ---")
    
    try:
        # Create conformal map from brain boundary
        def brain_boundary_func(t):
            """Parametric brain boundary function."""
            idx = int(t / (2*np.pi) * len(brain_boundary)) % len(brain_boundary)
            return complex(brain_boundary[idx, 0], brain_boundary[idx, 1])
        
        brain_map = MFSConformalMap(brain_boundary_func, n_boundary=100, n_charge=80)
        
        # Forward solver
        conformal_forward = ConformalForwardSolver(brain_map, n_sensors)
        u_conformal = conformal_forward.solve(sources)
        np.random.seed(seed)
        u_conformal_noisy = u_conformal + noise_level * np.std(u_conformal) * np.random.randn(len(u_conformal))
        
        # Linear solver
        conformal_linear = ConformalLinearInverseSolver(
            brain_map, n_boundary=n_sensors,
            source_resolution=source_resolution,
            verbose=True
        )
        conformal_linear.build_greens_matrix()
        
        print(f"  Source grid points: {len(conformal_linear.interior_points)}")
        
        # L-curve analysis
        print("\n  L-curve analysis for Conformal...")
        for method in ['l2', 'l1', 'tv']:
            print(f"    Computing L-curve for {method.upper()}...")
            alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
                conformal_linear, u_conformal_noisy, method=method, alphas=alphas
            )
            
            save_path = os.path.join(output_dir, f'l_curve_brain_conformal_{method}.png')
            opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                     method, 'Brain (Conformal)', save_path)
            l_curve_data[f'conformal_{method}'] = {'optimal_alpha': opt_alpha}
            
            if solutions[opt_idx] is not None:
                results[f'Conformal_{method.upper()}'] = {
                    'grid_intensities': solutions[opt_idx],
                    'grid_positions': conformal_linear.interior_points,
                    'alpha': opt_alpha,
                    'time': 0.0
                }
        
        # Conformal Nonlinear
        print("\n  Running Conformal Nonlinear...")
        conformal_nonlinear = ConformalNonlinearInverseSolver(brain_map, n_sources, n_sensors)
        
        t0 = time()
        sources_rec, residual = conformal_nonlinear.solve(u_conformal_noisy, method='SLSQP', n_restarts=10)
        elapsed = time() - t0
        
        rmse = compute_position_rmse(sources, sources_rec)
        results['Conformal_Nonlinear'] = {
            'sources': sources_rec,
            'rmse': rmse,
            'time': elapsed
        }
        print(f"    RMSE: {rmse:.2e}, Time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"  Conformal methods failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================================================================
    # FEM METHODS (polygon mesh via gmsh)
    # =========================================================================
    print("\n--- FEM Methods ---")
    
    # FEM Linear (uses gmsh for polygon/brain mesh)
    fem_linear = FEMLinearInverseSolver.from_polygon(
        vertices,
        forward_resolution=forward_resolution,
        source_resolution=source_resolution,
        verbose=True
    )
    fem_linear.build_greens_matrix(verbose=True)
    
    print(f"  FEM mesh nodes: {len(fem_linear.nodes)}")
    print(f"  FEM source grid points: {len(fem_linear.interior_points)}")
    
    # Generate measurements
    mesh_data = (fem_linear.nodes, fem_linear.elements, fem_linear.boundary_indices,
                 np.setdiff1d(np.arange(len(fem_linear.nodes)), fem_linear.boundary_indices))
    fem_forward = FEMForwardSolver(resolution=forward_resolution, verbose=False, mesh_data=mesh_data)
    u_fem = fem_forward.solve(sources)
    np.random.seed(seed)
    u_fem_noisy = u_fem + noise_level * np.std(u_fem) * np.random.randn(len(u_fem))
    
    # L-curve for FEM
    print("\n  L-curve analysis for FEM...")
    for method in ['l2', 'l1', 'tv']:
        print(f"    Computing L-curve for {method.upper()}...")
        alphas_out, residuals, regularizers, opt_idx, solutions = compute_l_curve_proper(
            fem_linear, u_fem_noisy, method=method, alphas=alphas
        )
        
        save_path = os.path.join(output_dir, f'l_curve_brain_fem_{method}.png')
        opt_alpha = plot_l_curve(alphas_out, residuals, regularizers, opt_idx,
                                 method, 'Brain (FEM)', save_path)
        l_curve_data[f'fem_{method}'] = {'optimal_alpha': opt_alpha}
        
        if solutions[opt_idx] is not None:
            results[f'FEM_{method.upper()}'] = {
                'grid_intensities': solutions[opt_idx],
                'grid_positions': fem_linear.interior_points,
                'alpha': opt_alpha,
                'time': 0.0
            }
    
    # FEM Nonlinear
    print("\n  Running FEM Nonlinear...")
    fem_nonlinear = FEMNonlinearInverseSolver.from_polygon(
        vertices, n_sources,
        resolution=forward_resolution,
        n_sensors=n_sensors,
        verbose=False
    )
    u_fem_nl = fem_nonlinear.forward.solve_at_sensors(sources)
    np.random.seed(seed)
    u_fem_nl_noisy = u_fem_nl + noise_level * np.std(u_fem_nl) * np.random.randn(len(u_fem_nl))
    fem_nonlinear.set_measured_data(u_fem_nl_noisy)
    
    t0 = time()
    result_fem_nl = fem_nonlinear.solve(method='SLSQP', n_restarts=10, maxiter=5000)
    elapsed = time() - t0
    
    rmse_fem = compute_position_rmse(sources, result_fem_nl.sources)
    results['FEM_Nonlinear'] = {
        'sources': result_fem_nl.sources,
        'rmse': rmse_fem,
        'time': elapsed
    }
    print(f"    RMSE: {rmse_fem:.2e}, Time: {elapsed:.2f}s")
    
    # Create comparison plot
    create_comparison_plot_custom(sources, results, 'brain', output_dir,
                                  domain_params={'boundary': brain_boundary})
    
    save_summary(results, l_curve_data, 'brain', output_dir)
    
    return results, l_curve_data


def compute_position_rmse(sources_true, sources_rec):
    """Compute position RMSE using optimal assignment."""
    from scipy.optimize import linear_sum_assignment
    
    n = len(sources_true)
    if len(sources_rec) == 0:
        return float('inf')
    
    m = len(sources_rec)
    cost = np.zeros((n, max(n, m)))
    
    for i, src_t in enumerate(sources_true):
        pos_t = src_t[0]
        for j, src_r in enumerate(sources_rec):
            if hasattr(src_r, 'position'):
                pos_r = src_r.position
            elif hasattr(src_r, 'x'):
                pos_r = (src_r.x, src_r.y)
            elif isinstance(src_r, tuple) and len(src_r) == 2 and isinstance(src_r[0], tuple):
                pos_r = src_r[0]
            else:
                pos_r = (src_r[0], src_r[1])
            
            cost[i, j] = (pos_t[0] - pos_r[0])**2 + (pos_t[1] - pos_r[1])**2
    
    row_ind, col_ind = linear_sum_assignment(cost[:n, :min(n, m)])
    return np.sqrt(cost[row_ind, col_ind].mean())


def create_comparison_plot_custom(sources_true, results, domain_type, output_dir, domain_params=None):
    """Create comprehensive comparison plot."""
    
    n_methods = len(results)
    if n_methods == 0:
        print("  No results to plot!")
        return
        
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Get domain boundary
    if domain_type == 'disk':
        theta = np.linspace(0, 2*np.pi, 100)
        boundary_x = np.cos(theta)
        boundary_y = np.sin(theta)
    elif domain_type == 'ellipse':
        a = domain_params.get('a', 1.5)
        b = domain_params.get('b', 0.8)
        theta = np.linspace(0, 2*np.pi, 100)
        boundary_x = a * np.cos(theta)
        boundary_y = b * np.sin(theta)
    elif domain_type == 'brain':
        boundary = domain_params.get('boundary')
        boundary_x = np.append(boundary[:, 0], boundary[0, 0])
        boundary_y = np.append(boundary[:, 1], boundary[0, 1])
    
    # Extract true source info
    true_x = [s[0][0] for s in sources_true]
    true_y = [s[0][1] for s in sources_true]
    true_q = [s[1] for s in sources_true]
    
    for idx, (name, data) in enumerate(results.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # Plot boundary
        ax.plot(boundary_x, boundary_y, 'k-', linewidth=1.5)
        
        # Plot true sources
        for i, (x, y, q) in enumerate(zip(true_x, true_y, true_q)):
            color = 'red' if q > 0 else 'blue'
            ax.scatter(x, y, s=200, marker='*', facecolors='none', edgecolors=color, linewidths=2, zorder=10)
        
        if 'sources' in data:
            # Nonlinear result
            rec_sources = data['sources']
            for src in rec_sources:
                if hasattr(src, 'position'):
                    rx, ry = src.position
                    rq = src.intensity
                elif hasattr(src, 'x'):
                    rx, ry = src.x, src.y
                    rq = src.intensity
                else:
                    rx, ry = src[0]
                    rq = src[1]
                
                color = 'lime' if rq > 0 else 'cyan'
                ax.scatter(rx, ry, c=color, s=100, marker='o', edgecolors='black', linewidths=1, zorder=9)
            
            title = f"{name}\nRMSE: {data['rmse']:.2e}"
        
        elif 'grid_intensities' in data:
            # Linear result
            q = data['grid_intensities']
            pos = data['grid_positions']
            
            vmax = np.max(np.abs(q)) if np.max(np.abs(q)) > 0 else 1
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=q, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax, shrink=0.7)
            
            title = f"{name}\nα={data.get('alpha', 'N/A'):.2e}"
        
        ax.set_xlim(boundary_x.min() - 0.1, boundary_x.max() + 0.1)
        ax.set_ylim(boundary_y.min() - 0.1, boundary_y.max() + 0.1)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_methods, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{domain_type.upper()} Domain - Method Comparison', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'comparison_{domain_type}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {save_path}")


def save_summary(results, l_curve_data, domain_type, output_dir):
    """Save summary to text file."""
    lines = []
    lines.append("="*60)
    lines.append(f"{domain_type.upper()} DOMAIN RESULTS")
    lines.append("="*60)
    lines.append("")
    lines.append(f"{'Method':<25} {'Metric':<20} {'Time (s)':<10}")
    lines.append("-"*60)
    
    for name, data in results.items():
        if 'rmse' in data:
            metric = f"RMSE = {data['rmse']:.4e}"
        elif 'alpha' in data:
            metric = f"α = {data['alpha']:.2e}"
        else:
            metric = "N/A"
        
        time_str = f"{data.get('time', 0):.2f}"
        lines.append(f"{name:<25} {metric:<20} {time_str:<10}")
    
    lines.append("")
    lines.append("L-curve optimal alphas:")
    for key, val in l_curve_data.items():
        lines.append(f"  {key}: α = {val['optimal_alpha']:.2e}")
    
    summary = "\n".join(lines)
    
    save_path = os.path.join(output_dir, f'summary_{domain_type}.txt')
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"\nSaved summary: {save_path}")
    print("\n" + summary)


def main():
    parser = argparse.ArgumentParser(description='Run inverse source comparison')
    parser.add_argument('--shape', type=str, default='all',
                        choices=['disk', 'ellipse', 'brain', 'all'],
                        help='Domain shape to run comparison for')
    parser.add_argument('--n_sources', type=int, default=4,
                        help='Number of sources')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--forward_res', type=float, default=0.05,
                        help='Forward mesh resolution')
    parser.add_argument('--source_res', type=float, default=0.1,
                        help='Source grid resolution')
    
    args = parser.parse_args()
    
    shapes_to_run = ['disk', 'ellipse', 'brain'] if args.shape == 'all' else [args.shape]
    
    for shape in shapes_to_run:
        output_dir = create_output_dir(shape)
        print(f"\n{'#'*70}")
        print(f"Running {shape.upper()} comparison")
        print(f"Output: {output_dir}")
        print(f"{'#'*70}")
        
        if shape == 'disk':
            run_disk_comparison(
                output_dir, 
                n_sources=args.n_sources,
                noise_level=args.noise,
                seed=args.seed,
                forward_resolution=args.forward_res,
                source_resolution=args.source_res
            )
        elif shape == 'ellipse':
            run_ellipse_comparison(
                output_dir,
                n_sources=args.n_sources,
                noise_level=args.noise,
                seed=args.seed,
                forward_resolution=args.forward_res,
                source_resolution=args.source_res
            )
        elif shape == 'brain':
            run_brain_comparison(
                output_dir,
                n_sources=args.n_sources,
                noise_level=args.noise,
                seed=args.seed,
                forward_resolution=args.forward_res,
                source_resolution=args.source_res
            )
    
    print("\n" + "="*70)
    print("ALL COMPARISONS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
