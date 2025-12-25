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
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Package information')
    
    return parser


def sources_to_tuples(sources):
    """Convert Source objects to ((x, y), q) tuples."""
    result = []
    for s in sources:
        if hasattr(s, 'x'):
            # Source object
            result.append(((s.x, s.y), s.intensity))
        else:
            # Already a tuple
            result.append(s)
    return result


def run_demo(args):
    """Run demonstration."""
    from . import bem_solver, conformal_bem
    from .utils import create_test_sources, plot_recovery_comparison
    import matplotlib.pyplot as plt
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.type in ('bem', 'all'):
        print("\n" + "="*60)
        print("BEM DEMONSTRATION")
        print("="*60)
        
        # Create sources
        sources_true = [
            ((-0.3, 0.4), 1.0),
            ((0.5, 0.3), 1.0),
            ((-0.4, -0.4), -1.0),
            ((0.3, -0.5), -1.0),
        ]
        
        # Forward solve
        forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
        u_measured = forward.solve(sources_true)
        u_measured += 0.001 * np.random.randn(len(u_measured))
        
        # Nonlinear inverse
        print("\nNonlinear solver...")
        inverse = bem_solver.BEMNonlinearInverseSolver(n_sources=4, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', maxiter=100)
        
        # Convert Source objects to tuples for compatibility
        sources_rec = sources_to_tuples(result.sources)
        
        # Compute recovered boundary data
        u_rec = forward.solve(sources_rec)
        
        print("\nTrue sources:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        print("\nRecovered sources:")
        for i, ((x, y), q) in enumerate(sources_rec):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.3f}")
        
        # Plot
        theta = forward.theta
        fig = plot_recovery_comparison(sources_true, sources_rec, theta, u_measured, u_rec)
        fig.savefig(output_dir / 'bem_demo.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_dir / 'bem_demo.png'}")
        
        if args.type != 'all':
            plt.show()
    
    if args.type in ('conformal', 'all'):
        print("\n" + "="*60)
        print("CONFORMAL BEM DEMONSTRATION (ELLIPSE)")
        print("="*60)
        
        ellipse = conformal_bem.EllipseMap(a=2.0, b=1.0)
        
        sources_true = [
            ((-0.5, 0.3), 1.0),
            ((0.8, 0.2), 1.0),
            ((-0.3, -0.3), -1.0),
            ((0.5, -0.4), -1.0),
        ]
        
        solver = conformal_bem.ConformalBEMSolver(ellipse, n_boundary=100)
        u_measured = solver.solve_forward(sources_true)
        u_measured += 0.001 * np.random.randn(len(u_measured))
        
        inverse = conformal_bem.ConformalNonlinearInverse(ellipse, n_sources=4, n_boundary=100)
        inverse.set_measured_data(u_measured)
        result = inverse.solve(method='L-BFGS-B', maxiter=100)
        sources_rec = sources_to_tuples(result.sources)
        
        print("\nTrue sources:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.2f}")
        
        print("\nRecovered sources:")
        for i, ((x, y), q) in enumerate(sources_rec):
            print(f"  {i+1}: ({x:+.3f}, {y:+.3f}), q = {q:+.3f}")
        
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
    from . import bem_solver
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
    forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
    u_clean = forward.solve(sources_true)
    u_measured = u_clean + args.noise * np.random.randn(len(u_clean))
    
    print(f"\nAdded noise with σ = {args.noise}")
    
    # Linear inverse solve
    print(f"\nSolving with {args.method.upper()} regularization...")
    
    linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
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
    from . import bem_solver
    from .parameter_study import l_curve_analysis, plot_l_curve_comparison, save_results
    from .utils import create_test_sources
    
    print("Setting up sweep...")
    
    # Create test problem
    sources_true = create_test_sources(4, seed=42)
    
    forward = bem_solver.BEMForwardSolver(n_boundary_points=100)
    u_measured = forward.solve(sources_true)
    u_measured += 0.001 * np.random.randn(len(u_measured))
    
    linear = bem_solver.BEMLinearInverseSolver(n_boundary=100)
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
  - Conformal BEM for general simply connected domains
  - FEM (Finite Element Method) fallback

Regularization:
  - L2 (Tikhonov) - smooth solutions
  - L1 (Sparsity) - sparse solutions via IRLS
  - TV (Total Variation) - piecewise constant via Chambolle-Pock or ADMM

Domain Support:
  - Unit disk (analytical)
  - Ellipse (Joukowsky map)
  - Star-shaped domains (numerical conformal map)

For more information:
  - Documentation: docs/main.pdf
  - Examples: examples/complete_example.py
  - GitHub: https://github.com/Shaerdan/inverse_source_project
    """)


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
    elif args.command == 'info':
        run_info(args)


if __name__ == '__main__':
    main()
