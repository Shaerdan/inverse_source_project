#!/usr/bin/env python
"""
Command-Line Interface for Inverse Source Localization
=======================================================

Usage:
    python -m inverse_source.cli --help
    python -m inverse_source.cli solve --method l1 --alpha 1e-4
    python -m inverse_source.cli sweep --output results/sweep.json
    python -m inverse_source.cli demo --type bem
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path


def setup_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog='inverse_source',
        description='Inverse Source Localization Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a demo
  python -m inverse_source.cli demo --type bem
  
  # Solve with L1 regularization
  python -m inverse_source.cli solve --method l1 --alpha 1e-4
  
  # Parameter sweep
  python -m inverse_source.cli sweep --method l1 --output sweep.json
  
  # Create config file
  python -m inverse_source.cli config --template default --output config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    demo_parser.add_argument('--type', choices=['bem', 'conformal', 'comparison', 'all'],
                            default='all', help='Demo type')
    demo_parser.add_argument('--output-dir', type=str, default='results',
                            help='Output directory for figures')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve inverse problem')
    solve_parser.add_argument('--method', choices=['l1', 'l2', 'tv'],
                             default='l1', help='Regularization method')
    solve_parser.add_argument('--alpha', type=float, default=None,
                             help='Regularization parameter (auto if not given)')
    solve_parser.add_argument('--n-sources', type=int, default=4,
                             help='Number of sources (for synthetic data)')
    solve_parser.add_argument('--noise', type=float, default=0.001,
                             help='Noise level for synthetic data')
    solve_parser.add_argument('--domain', choices=['disk', 'ellipse', 'star'],
                             default='disk', help='Domain type')
    solve_parser.add_argument('--config', type=str, default=None,
                             help='Path to config file')
    solve_parser.add_argument('--output', type=str, default=None,
                             help='Output file for results')
    solve_parser.add_argument('--plot', action='store_true',
                             help='Show plots')
    
    # Sweep command
    sweep_parser = subparsers.add_parser('sweep', help='Parameter sweep')
    sweep_parser.add_argument('--method', choices=['l1', 'l2', 'tv', 'all'],
                             default='all', help='Regularization method(s)')
    sweep_parser.add_argument('--alpha-min', type=float, default=1e-6,
                             help='Minimum alpha')
    sweep_parser.add_argument('--alpha-max', type=float, default=1e-1,
                             help='Maximum alpha')
    sweep_parser.add_argument('--n-alpha', type=int, default=30,
                             help='Number of alpha values')
    sweep_parser.add_argument('--output', type=str, default='sweep_results.json',
                             help='Output file')
    sweep_parser.add_argument('--plot', action='store_true',
                             help='Show L-curve plot')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--template', type=str, default='default',
                              help='Template name')
    config_parser.add_argument('--output', type=str, default='config.json',
                              help='Output config file')
    config_parser.add_argument('--list-templates', action='store_true',
                              help='List available templates')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare all solvers')
    compare_parser.add_argument('--quick', action='store_true', 
                               help='Quick mode (fixed α, no sweep)')
    compare_parser.add_argument('--optimal', action='store_true',
                               help='Use optimal α for each method (via L-curve)')
    compare_parser.add_argument('--methods', type=str, nargs='+',
                               default=['l1', 'l2', 'tv_admm', 'tv_cp'],
                               help='Regularization methods to compare')
    compare_parser.add_argument('--noise', type=float, default=0.001,
                               help='Noise level')
    compare_parser.add_argument('--alpha', type=float, default=1e-4,
                               help='Regularization parameter (for quick mode)')
    compare_parser.add_argument('--seed', type=int, default=42,
                               help='Random seed for reproducibility')
    compare_parser.add_argument('--no-nonlinear', action='store_true',
                               help='Skip nonlinear solvers')
    compare_parser.add_argument('--output-dir', type=str, default='results',
                               help='Output directory for results')
    compare_parser.add_argument('--save', type=str, default=None,
                               help='Save figure to path (overrides auto-naming)')
    compare_parser.add_argument('--no-plot', action='store_true',
                               help='Skip plotting')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Package information')
    
    return parser


def run_demo(args):
    """Run demonstration."""
    try:
        from . import analytical_solver, conformal_solver
    except ImportError:
        import analytical_solver, conformal_solver
    from .utils import create_test_sources, plot_recovery_comparison
    import matplotlib.pyplot as plt
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.type in ('bem', 'analytical', 'all'):
        print("\n" + "="*60)
        print("ANALYTICAL SOLVER DEMONSTRATION")
        print("="*60)
        
        # Create sources
        sources_true = [
            ((-0.3, 0.4), 1.0),
            ((0.5, 0.3), 1.0),
            ((-0.4, -0.4), -1.0),
            ((0.3, -0.5), -1.0),
        ]
        
        # Forward solve
        forward = analytical_solver.AnalyticalForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        u_measured += 0.001 * np.random.randn(len(u_measured))
        
        # Nonlinear inverse
        print("\nNonlinear solver...")
        inverse = analytical_solver.AnalyticalNonlinearInverseSolver(n_sources=4, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', maxiter=100)
        sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
        
        # Compute recovered boundary data
        u_rec = forward.solve(sources_rec)
        
        print("\nTrue sources:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        print("\nRecovered sources:")
        for i, ((x, y), q) in enumerate(sources_rec):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        # Plot
        theta = forward.theta
        fig = plot_recovery_comparison(sources_true, sources_rec, theta, u_measured, u_rec)
        fig.savefig(output_dir / 'analytical_demo.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir / 'analytical_demo.png'}")
        
        if args.type != 'all':
            plt.show()
    
    if args.type in ('conformal', 'all'):
        print("\n" + "="*60)
        print("CONFORMAL SOLVER DEMONSTRATION (ELLIPSE)")
        print("="*60)
        
        ellipse = conformal_solver.EllipseMap(a=2.0, b=1.0)
        
        sources_true = [
            ((-0.5, 0.3), 1.0),
            ((0.8, 0.2), 1.0),
            ((-0.3, -0.3), -1.0),
            ((0.5, -0.4), -1.0),
        ]
        
        solver = conformal_solver.ConformalForwardSolver(ellipse, n_boundary=100)
        u_measured = solver.solve(sources_true)
        u_measured += 0.001 * np.random.randn(len(u_measured))
        
        inverse = conformal_solver.ConformalNonlinearInverseSolver(ellipse, n_sources=4, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', maxiter=100)
        sources_rec = [((s.x, s.y), s.intensity) for s in result.sources]
        
        print("\nTrue sources:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        print("\nRecovered sources:")
        for i, ((x, y), q) in enumerate(sources_rec):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        boundary = ellipse.boundary_physical(200)
        ax.plot(np.real(boundary), np.imag(boundary), 'k-', linewidth=2)
        
        for (x, y), q in sources_true:
            color = 'red' if q > 0 else 'blue'
            ax.plot(x, y, 'o', color=color, markersize=15, 
                    markerfacecolor='none', markeredgewidth=2)
        
        for (x, y), q in sources_rec:
            color = 'red' if q > 0 else 'blue'
            ax.plot(x, y, '+', color=color, markersize=15, markeredgewidth=3)
        
        ax.set_aspect('equal')
        ax.set_title('Ellipse Domain: True (○) vs Recovered (+)')
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_dir / 'conformal_demo.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir / 'conformal_demo.png'}")
        
        if args.type != 'all':
            plt.show()
    
    if args.type == 'all':
        plt.show()
    
    print("\nDemo complete!")


def run_solve(args):
    """Run inverse problem solver."""
    try:
        from . import analytical_solver
    except ImportError:
        import analytical_solver
    from .config import get_config
    from .utils import create_test_sources, compute_source_error
    
    # Load config if provided
    if args.config:
        config = get_config(args.config)
    else:
        from .config import Config
        config = Config()
    
    print("Setting up problem...")
    
    # Create test sources
    sources_true = create_test_sources(args.n_sources, seed=42)
    
    print(f"\nTrue sources ({args.n_sources}):")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.3f}")
    
    # Forward solve
    forward = analytical_solver.AnalyticalForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    u_measured = u_clean + args.noise * np.random.randn(len(u_clean))
    
    print(f"\nAdded noise with σ = {args.noise}")
    
    # Linear inverse solve
    print(f"\nSolving with {args.method.upper()} regularization...")
    
    linear = analytical_solver.AnalyticalLinearInverseSolver(n_boundary=100)
    linear.build_greens_matrix()
    
    u_centered = u_measured - np.mean(u_measured)
    
    if args.alpha is None:
        # Find optimal alpha
        from .parameter_study import parameter_sweep
        print("Finding optimal α...")
        sweep = parameter_sweep(linear, u_measured, method=args.method, verbose=False)
        alpha = sweep.alpha_optimal
        print(f"Optimal α = {alpha:.2e}")
    else:
        alpha = args.alpha
    
    # Solve
    if args.method == 'l1':
        q = linear.solve_l1(u_centered, alpha=alpha)
    elif args.method == 'l2':
        q = linear.solve_l2(u_centered, alpha=alpha)
    else:
        print("TV not implemented in CLI yet")
        return
    
    # Extract significant sources
    threshold = 0.1 * np.max(np.abs(q))
    significant_idx = np.where(np.abs(q) > threshold)[0]
    
    sources_rec = []
    for idx in significant_idx:
        pos = linear.interior_points[idx]
        sources_rec.append(((pos[0], pos[1]), q[idx]))
    
    print(f"\nRecovered {len(sources_rec)} significant sources:")
    for i, ((x, y), q_val) in enumerate(sources_rec):
        print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q_val:+.3f}")
    
    # Compute errors
    metrics = compute_source_error(sources_true, sources_rec)
    print(f"\nMetrics:")
    print(f"  Position RMSE: {metrics['position_rmse']:.4f}")
    print(f"  Intensity RMSE: {metrics['intensity_rmse']:.4f}")
    
    # Save results
    if args.output:
        results = {
            'sources_true': sources_true,
            'sources_recovered': sources_rec,
            'alpha': alpha,
            'method': args.method,
            'metrics': metrics,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, np.ndarray) else x)
        print(f"\nResults saved to {args.output}")
    
    # Plot
    if args.plot:
        from .utils import plot_recovery_comparison
        import matplotlib.pyplot as plt
        
        theta = forward.theta
        u_rec = linear.G @ q + np.mean(u_measured)
        fig = plot_recovery_comparison(sources_true, sources_rec, theta, u_measured, u_rec)
        plt.show()


def run_sweep(args):
    """Run parameter sweep."""
    try:
        from . import analytical_solver
    except ImportError:
        import analytical_solver
    from .parameter_study import l_curve_analysis, plot_l_curve_comparison, save_results
    from .utils import create_test_sources
    
    print("Setting up sweep...")
    
    # Create test problem
    sources_true = create_test_sources(4, seed=42)
    
    forward = analytical_solver.AnalyticalForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    linear = analytical_solver.AnalyticalLinearInverseSolver(n_boundary=100)
    linear.build_greens_matrix()
    
    # Create alpha range
    alphas = np.logspace(np.log10(args.alpha_min), np.log10(args.alpha_max), args.n_alpha)
    
    # Run analysis
    if args.method == 'all':
        methods = ['l1', 'l2']
    else:
        methods = [args.method]
    
    results = l_curve_analysis(linear, u_measured, methods=methods, alphas=alphas)
    
    # Save results
    save_results(results, args.output)
    
    # Plot
    if args.plot:
        import matplotlib.pyplot as plt
        plot_l_curve_comparison(results)
        plt.show()


def run_config(args):
    """Handle config commands."""
    from .config import TEMPLATES, get_template
    
    if args.list_templates:
        print("Available templates:")
        for name in TEMPLATES.keys():
            print(f"  - {name}")
        return
    
    config = get_template(args.template)
    config.save(args.output)


def run_info(args):
    """Show package information."""
    print("""
Inverse Source Localization Toolkit
===================================

Methods:
  - BEM (Boundary Element Method) with analytical Green's function
  - FEM (Finite Element Method) with uniform triangular mesh
  - Conformal BEM for general simply connected domains

Forward Methods:
  - BEM: Analytical Green's function (fast, exact on unit disk)
  - FEM: Finite element discretization (mesh-based)

Inverse Methods (Linear/Distributed):
  - L2 (Tikhonov): Smooth solutions, closed-form
  - L1 (Sparsity): Sparse solutions via IRLS
  - TV-ADMM: Piecewise constant via ADMM
  - TV-CP: Piecewise constant via Chambolle-Pock

Inverse Methods (Nonlinear/Continuous):
  - L-BFGS-B: Local optimizer (fast, may get stuck)
  - differential_evolution: Global optimizer (slower, more robust)
  - basinhopping: Global with local polish

Domain Support:
  - Unit disk (analytical)
  - Ellipse (Joukowsky map)
  - Star-shaped domains (numerical conformal map)

For more information:
  - Documentation: docs/main.pdf
  - Examples: examples/complete_example.py
  - GitHub: https://github.com/Shaerdan/inverse_source_project
    """)


def run_compare(args):
    """Run solver comparison with results tracking."""
    from .comparison import (compare_all_solvers, compare_with_optimal_alpha, 
                            print_comparison_table, plot_comparison)
    import matplotlib.pyplot as plt
    import hashlib
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Test sources
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    # Create hash from key parameters
    param_str = f"seed={args.seed}_noise={args.noise}_optimal={args.optimal}"
    if not args.optimal:
        param_str += f"_alpha={args.alpha}"
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"run_{param_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    params = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'noise_level': args.noise,
        'optimal_alpha': args.optimal,
        'alpha': args.alpha if not args.optimal else 'auto',
        'methods': args.methods,
        'include_nonlinear': not args.no_nonlinear,
        'sources_true': sources_true,
        'param_hash': param_hash,
    }
    
    params_file = output_dir / 'params.json'
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    print("="*70)
    print("INVERSE SOURCE LOCALIZATION - SOLVER COMPARISON")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Parameter hash: {param_hash}")
    print(f"Seed: {args.seed}")
    print(f"True sources: {len(sources_true)}")
    print(f"Noise level: {args.noise}")
    
    if args.optimal:
        print(f"Methods: {args.methods}")
        print("Mode: Optimal α selection via L-curve")
        
        results = compare_with_optimal_alpha(
            sources_true,
            noise_level=args.noise,
            methods=args.methods,
            include_nonlinear=not args.no_nonlinear,
            seed=args.seed,
            verbose=True
        )
    elif args.quick:
        print(f"Alpha (linear): {args.alpha}")
        print("Mode: Quick (fixed α)")
        
        results = compare_all_solvers(
            sources_true,
            noise_level=args.noise,
            alpha_linear=args.alpha,
            quick=True,
            seed=args.seed
        )
    else:
        print(f"Alpha (linear): {args.alpha}")
        print("Mode: Full comparison (fixed α)")
        
        results = compare_all_solvers(
            sources_true,
            noise_level=args.noise,
            alpha_linear=args.alpha,
            quick=False,
            seed=args.seed
        )
    
    # Print table
    print_comparison_table(results)
    
    # Save results to JSON
    results_data = []
    for r in results:
        results_data.append({
            'solver_name': r.solver_name,
            'method_type': r.method_type,
            'forward_type': r.forward_type,
            'position_rmse': r.position_rmse,
            'intensity_rmse': r.intensity_rmse,
            'boundary_residual': r.boundary_residual,
            'time_seconds': r.time_seconds,
            'iterations': r.iterations,
            'sources_recovered': r.sources_recovered,
        })
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Plot
    if not args.no_plot:
        if args.save:
            save_path = args.save
        else:
            save_path = output_dir / 'comparison.png'
        
        fig = plot_comparison(results, sources_true, save_path=str(save_path))
        print(f"Figure saved to: {save_path}")
        plt.show()


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'demo':
        run_demo(args)
    elif args.command == 'solve':
        run_solve(args)
    elif args.command == 'sweep':
        run_sweep(args)
    elif args.command == 'config':
        run_config(args)
    elif args.command == 'compare':
        run_compare(args)
    elif args.command == 'info':
        run_info(args)


if __name__ == '__main__':
    main()
