#!/usr/bin/env python
"""
Command-Line Interface for Inverse Source Localization
=======================================================

Usage:
    python -m inverse_source --help
    python -m inverse_source compare --domains disk ellipse star
    python -m inverse_source demo --type bem
    
    # Or run directly:
    cd src && python cli.py compare --domains disk ellipse
"""

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path

# Import compatibility: handle both package and direct execution
def _import_module(name):
    """Import module with fallback from relative to absolute import."""
    try:
        return __import__(f".{name}", globals(), locals(), [name], 1)
    except ImportError:
        return __import__(name)

# Check if running as package
_IS_PACKAGE = __name__ != "__main__" or __package__ is not None


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
                               help='Quick mode (skip slow solvers)')
    compare_parser.add_argument('--fixed-alpha', action='store_true',
                               help='Use fixed α instead of L-curve optimal')
    compare_parser.add_argument('--use-calibration', type=str, default=None,
                               help='Path to calibration config JSON (uses calibrated parameters)')
    compare_parser.add_argument('--no-calibration', action='store_true',
                               help='Skip auto-loading calibration even if available')
    compare_parser.add_argument('--methods', type=str, nargs='+',
                               default=['l1', 'l2', 'tv_admm', 'tv_cp'],
                               help='Regularization methods to compare')
    compare_parser.add_argument('--noise', type=float, default=0.001,
                               help='Noise level')
    compare_parser.add_argument('--alpha', type=float, default=1e-4,
                               help='Regularization parameter (only with --fixed-alpha)')
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
    compare_parser.add_argument('--domain', type=str, default='disk',
                               choices=['disk', 'ellipse', 'star', 'square', 'polygon', 'brain'],
                               help='Domain type (default: disk). Use --domains for multiple.')
    compare_parser.add_argument('--domains', type=str, nargs='+', default=None,
                               help='Multiple domains to compare (e.g., --domains disk ellipse star)')
    compare_parser.add_argument('--ellipse-a', type=float, default=2.0,
                               help='Ellipse semi-major axis (for --domain ellipse)')
    compare_parser.add_argument('--ellipse-b', type=float, default=1.0,
                               help='Ellipse semi-minor axis (for --domain ellipse)')
    compare_parser.add_argument('--vertices', type=str, default=None,
                               help='Polygon vertices as JSON, e.g., "[[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]]"')
    
    # Calibrate command
    cal_parser = subparsers.add_parser('calibrate', help='Calibrate parameters for all domains')
    cal_parser.add_argument('--domains', type=str, nargs='+', default=None,
                           help='Domains to calibrate (default: all)')
    cal_parser.add_argument('--output-dir', type=str, default='results/calibration',
                           help='Output directory for calibration results')
    cal_parser.add_argument('--noise', type=float, default=0.001,
                           help='Noise level for calibration')
    cal_parser.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    cal_parser.add_argument('--plot', action='store_true',
                           help='Generate plots after calibration')
    
    # Convergence command
    conv_parser = subparsers.add_parser('convergence', help='Run mesh convergence study')
    conv_parser.add_argument('--domain', type=str, default='disk',
                            choices=['disk', 'ellipse', 'star', 'square', 'polygon', 'brain'],
                            help='Domain type')
    conv_parser.add_argument('--forward-only', action='store_true',
                            help='Run only forward mesh convergence')
    conv_parser.add_argument('--inverse-only', action='store_true',
                            help='Run only inverse source grid convergence')
    conv_parser.add_argument('--output-dir', type=str, default='results/convergence',
                            help='Output directory for results')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Package information')
    
    return parser


def run_demo(args):
    """Run demonstration."""
    try:
        from . import analytical_solver, conformal_solver
        from .utils import create_test_sources, plot_recovery_comparison
    except ImportError:
        import analytical_solver, conformal_solver
        from utils import create_test_sources, plot_recovery_comparison
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
        from .config import get_config, Config
        from .utils import create_test_sources, compute_source_error
        from .parameter_study import parameter_sweep
    except ImportError:
        import analytical_solver
        from config import get_config, Config
        from utils import create_test_sources, compute_source_error
        from parameter_study import parameter_sweep
    
    # Load config if provided
    if args.config:
        config = get_config(args.config)
    else:
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
        try:
            from .utils import plot_recovery_comparison
        except ImportError:
            from utils import plot_recovery_comparison
        import matplotlib.pyplot as plt
        
        theta = forward.theta
        u_rec = linear.G @ q + np.mean(u_measured)
        fig = plot_recovery_comparison(sources_true, sources_rec, theta, u_measured, u_rec)
        plt.show()


def run_sweep(args):
    """Run parameter sweep."""
    try:
        from . import analytical_solver
        from .parameter_study import l_curve_analysis, plot_l_curve_comparison, save_results
        from .utils import create_test_sources
    except ImportError:
        import analytical_solver
        from parameter_study import l_curve_analysis, plot_l_curve_comparison, save_results
        from utils import create_test_sources
    
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
    try:
        from .config import TEMPLATES, get_template
    except ImportError:
        from config import TEMPLATES, get_template
    
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
    """Run solver comparison with experiment tracking."""
    try:
        from .comparison import (compare_all_solvers, compare_with_optimal_alpha, 
                                print_comparison_table, plot_comparison,
                                run_domain_comparison, create_domain_sources,
                                compare_all_solvers_general)
        from .experiment_tracker import ExperimentTracker
    except ImportError:
        from comparison import (compare_all_solvers, compare_with_optimal_alpha, 
                               print_comparison_table, plot_comparison,
                               run_domain_comparison, create_domain_sources,
                               compare_all_solvers_general)
        from experiment_tracker import ExperimentTracker
    import matplotlib.pyplot as plt
    import json as json_module
    
    # Handle multiple domains
    if args.domains:
        run_compare_multi_domain(args)
        return
    
    # Single domain comparison (original logic)
    run_compare_single_domain(args)


def run_compare_multi_domain(args):
    """Run comparison across multiple domains."""
    try:
        from .comparison import (compare_all_solvers_general, print_comparison_table, 
                                plot_comparison, create_domain_sources)
    except ImportError:
        from comparison import (compare_all_solvers_general, print_comparison_table, 
                               plot_comparison, create_domain_sources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json as json_module
    from pathlib import Path
    from datetime import datetime
    
    DEFAULT_CALIBRATION_PATH = 'results/calibration/calibration_config.json'
    
    # Load calibration if available
    cal_config = None
    calibration_path = args.use_calibration or (
        DEFAULT_CALIBRATION_PATH if os.path.exists(DEFAULT_CALIBRATION_PATH) and not args.no_calibration else None
    )
    
    if calibration_path:
        try:
            from .calibration import load_calibration_config, get_domain_params
        except ImportError:
            from calibration import load_calibration_config, get_domain_params
        print(f"Loading calibration: {calibration_path}")
        cal_config = load_calibration_config(calibration_path)
    
    # Output directory
    output_dir = Path(args.output_dir) / f"compare_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE SOLVER COMPARISON - ALL DOMAINS")
    print("=" * 70)
    print(f"Domains: {args.domains}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print("=" * 70)
    
    all_results = {}
    
    for domain in args.domains:
        print(f"\n{'='*70}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*70}")
        
        # Get domain-specific params
        domain_params = None
        if domain == 'ellipse':
            domain_params = {'a': args.ellipse_a, 'b': args.ellipse_b}
        elif domain == 'square':
            domain_params = {'vertices': [(-1, -1), (1, -1), (1, 1), (-1, 1)]}
        
        # Get calibration for this domain
        calibration_params = None
        if cal_config:
            try:
                calibration_params = get_domain_params(cal_config, domain)
                if calibration_params:
                    print(f"  Using calibrated parameters for {domain}")
            except:
                pass
        
        # Get test sources
        sources_true = create_domain_sources(domain, domain_params)
        
        # Determine alpha
        if calibration_params:
            alpha_dict = {
                'l1': calibration_params['alpha_l1'],
                'l2': calibration_params['alpha_l2'],
                'tv': calibration_params['alpha_tv']
            }
            fwd_res = calibration_params.get('forward_mesh_resolution', 0.1)
            src_res = calibration_params.get('source_grid_resolution', 0.15)
        else:
            alpha_dict = args.alpha
            fwd_res = 0.1
            src_res = 0.15
        
        # Run comparison
        try:
            results = compare_all_solvers_general(
                domain_type=domain,
                domain_params=domain_params,
                sources_true=sources_true,
                noise_level=args.noise,
                alpha=alpha_dict if calibration_params else ('auto' if not args.fixed_alpha else args.alpha),
                forward_resolution=fwd_res,
                source_resolution=src_res,
                quick=args.quick,
                seed=args.seed,
                verbose=True
            )
            
            all_results[domain] = {
                'results': results,
                'sources_true': sources_true,
                'domain_params': domain_params
            }
            
            # Print table
            print_comparison_table(results)
            
            # Save individual domain plot
            if not args.no_plot:
                fig = plot_comparison(results, sources_true, 
                                     domain_type=domain, domain_params=domain_params)
                fig.savefig(output_dir / f'{domain}_comparison.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Figure saved: {domain}_comparison.png")
                
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'domains': args.domains,
        'mode': 'quick' if args.quick else 'full',
        'results': {}
    }
    
    for domain, data in all_results.items():
        summary['results'][domain] = []
        for r in data['results']:
            summary['results'][domain].append({
                'solver_name': r.solver_name,
                'method_type': r.method_type,
                'position_rmse': float(r.position_rmse),
                'intensity_rmse': float(r.intensity_rmse),
                'time_seconds': float(r.time_seconds),
                'sources_recovered': len(r.sources_recovered) if r.sources_recovered else 0
            })
    
    with open(output_dir / 'summary.json', 'w') as f:
        json_module.dump(summary, f, indent=2)
    
    # Create combined summary figure
    if not args.no_plot and all_results:
        create_combined_summary_figure(all_results, output_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - summary.json")
    for domain in all_results:
        print(f"  - {domain}_comparison.png")
    print(f"  - combined_summary.png")


def create_combined_summary_figure(all_results: dict, output_dir):
    """Create a summary figure comparing all domains."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    domains = list(all_results.keys())
    n_domains = len(domains)
    
    fig, axes = plt.subplots(2, n_domains, figsize=(5*n_domains, 8))
    if n_domains == 1:
        axes = axes.reshape(2, 1)
    
    for i, domain in enumerate(domains):
        results = all_results[domain]['results']
        
        # Top row: Position RMSE by solver
        ax = axes[0, i]
        solver_names = [r.solver_name[:20] for r in results]  # Truncate names
        pos_errors = [r.position_rmse for r in results]
        colors = ['green' if e < 0.1 else 'orange' if e < 0.5 else 'red' for e in pos_errors]
        
        bars = ax.barh(range(len(solver_names)), pos_errors, color=colors)
        ax.set_yticks(range(len(solver_names)))
        ax.set_yticklabels(solver_names, fontsize=7)
        ax.set_xlabel('Position RMSE')
        ax.set_title(f'{domain.upper()}\nPosition Error')
        ax.axvline(x=0.1, color='g', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.invert_yaxis()
        
        # Bottom row: Time by solver
        ax = axes[1, i]
        times = [r.time_seconds for r in results]
        ax.barh(range(len(solver_names)), times, color='steelblue')
        ax.set_yticks(range(len(solver_names)))
        ax.set_yticklabels(solver_names, fontsize=7)
        ax.set_xlabel('Time (s)')
        ax.set_title('Computation Time')
        ax.invert_yaxis()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'combined_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Combined summary saved: combined_summary.png")


def run_compare_single_domain(args):
    """Run comparison for a single domain (original logic)."""
    try:
        from .comparison import (compare_all_solvers, compare_with_optimal_alpha, 
                                print_comparison_table, plot_comparison,
                                run_domain_comparison, create_domain_sources,
                                compare_all_solvers_general)
        from .experiment_tracker import ExperimentTracker
    except ImportError:
        from comparison import (compare_all_solvers, compare_with_optimal_alpha, 
                               print_comparison_table, plot_comparison,
                               run_domain_comparison, create_domain_sources,
                               compare_all_solvers_general)
        from experiment_tracker import ExperimentTracker
    import matplotlib.pyplot as plt
    import json as json_module
    
    # Default calibration path
    DEFAULT_CALIBRATION_PATH = 'results/calibration/calibration_config.json'
    
    # Auto-detect calibration if not explicitly specified
    calibration_params = None
    calibration_path = None
    
    if args.use_calibration:
        # Explicit path provided
        calibration_path = args.use_calibration
    elif not getattr(args, 'no_calibration', False):
        # Check if default calibration exists
        if os.path.exists(DEFAULT_CALIBRATION_PATH):
            calibration_path = DEFAULT_CALIBRATION_PATH
            print(f"Found calibration config at: {calibration_path}")
            print(f"  (Use --no-calibration to skip, or --use-calibration PATH for different config)")
    
    if calibration_path:
        try:
            from .calibration import load_calibration_config, get_domain_params
        except ImportError:
            from calibration import load_calibration_config, get_domain_params
        
        print(f"Loading calibration: {calibration_path}")
        cal_config = load_calibration_config(calibration_path)
        calibration_params = get_domain_params(cal_config, args.domain)
        if calibration_params:
            print(f"  Forward mesh resolution: {calibration_params['forward_mesh_resolution']}")
            print(f"  Source grid resolution: {calibration_params['source_grid_resolution']}")
            print(f"  Alpha L1: {calibration_params['alpha_l1']:.2e}")
            print(f"  Alpha L2: {calibration_params['alpha_l2']:.2e}")
            print(f"  Alpha TV: {calibration_params['alpha_tv']:.2e}")
        else:
            print(f"  Warning: No calibration found for domain '{args.domain}', using defaults")
    
    # Domain parameters
    domain_params = None
    if args.domain == 'ellipse':
        domain_params = {'a': args.ellipse_a, 'b': args.ellipse_b}
    elif args.domain == 'polygon':
        if args.vertices:
            try:
                vertices = json_module.loads(args.vertices)
                domain_params = {'vertices': [tuple(v) for v in vertices]}
            except json_module.JSONDecodeError:
                print(f"Error: Invalid JSON for vertices: {args.vertices}")
                print("Example: --vertices '[[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]]'")
                return
        else:
            domain_params = {'vertices': [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]}
    
    # Override with calibration domain_params if available
    if calibration_params and calibration_params.get('domain_params'):
        domain_params = calibration_params['domain_params']
    
    # Get test sources
    sources_true = create_domain_sources(args.domain, domain_params)
    
    # Determine alpha source: calibration > fixed > L-curve
    if calibration_params:
        use_optimal = False
        use_calibration = True
        alpha_dict = {
            'l1': calibration_params['alpha_l1'],
            'l2': calibration_params['alpha_l2'],
            'tv': calibration_params['alpha_tv']
        }
        alpha_selection = 'calibrated'
    elif args.fixed_alpha:
        use_optimal = False
        use_calibration = False
        alpha_dict = None
        alpha_selection = 'fixed'
    else:
        use_optimal = True
        use_calibration = False
        alpha_dict = None
        alpha_selection = 'lcurve'
    
    # Initialize experiment tracker
    experiment_name = f"compare_{args.domain}"
    with ExperimentTracker(base_dir=args.output_dir, experiment_name=experiment_name) as tracker:
        
        # Log all configuration
        tracker.log_domain(args.domain, domain_params)
        tracker.log_sources(sources_true, noise_level=args.noise, seed=args.seed)
        tracker.log_solver(
            solver_type='comparison',
            method='all',
            alpha=args.alpha if not use_optimal and not use_calibration else 0.0,
            alpha_selection=alpha_selection,
            quick_mode=args.quick,
            include_nonlinear=not args.no_nonlinear,
            methods=args.methods
        )
        
        if calibration_params:
            tracker.log_mesh(
                forward_resolution=calibration_params.get('forward_mesh_resolution'),
                source_resolution=calibration_params.get('source_grid_resolution')
            )
            tracker.log_params(calibration_file=args.use_calibration)
        
        print("="*70)
        print("INVERSE SOURCE LOCALIZATION - COMPREHENSIVE SOLVER COMPARISON")
        print("="*70)
        print(f"\nExperiment ID: {tracker.experiment_id}")
        print(f"Output directory: {tracker.output_dir}")
        print(f"\nDomain: {args.domain}")
        if args.domain == 'ellipse':
            print(f"  Semi-axes: a={args.ellipse_a}, b={args.ellipse_b}")
        elif args.domain in ['polygon', 'square']:
            vertices = domain_params.get('vertices') if domain_params else None
            if vertices:
                print(f"  Vertices: {len(vertices)} points")
        print(f"Seed: {args.seed}")
        print(f"True sources: {len(sources_true)}")
        print(f"Noise level: {args.noise}")
        print(f"Alpha selection: {alpha_selection}")
        print(f"Mode: {'Quick' if args.quick else 'Full'}")
        
        # Get mesh resolutions (from calibration or defaults)
        fwd_resolution = calibration_params.get('forward_mesh_resolution', 0.1) if calibration_params else 0.1
        src_resolution = calibration_params.get('source_grid_resolution', 0.15) if calibration_params else 0.15
        
        # Run comparison
        if use_calibration:
            print(f"\nUsing calibrated parameters...")
            results = compare_all_solvers_general(
                domain_type=args.domain,
                domain_params=domain_params,
                sources_true=sources_true,
                noise_level=args.noise,
                alpha=alpha_dict,
                forward_resolution=fwd_resolution,
                source_resolution=src_resolution,
                quick=args.quick,
                seed=args.seed,
                verbose=True
            )
        elif use_optimal:
            print(f"\nUsing L-curve analysis to find optimal α for each method...")
            results = compare_all_solvers_general(
                domain_type=args.domain,
                domain_params=domain_params,
                sources_true=sources_true,
                noise_level=args.noise,
                alpha='auto',
                forward_resolution=fwd_resolution,
                source_resolution=src_resolution,
                quick=args.quick,
                seed=args.seed,
                verbose=True
            )
        else:
            results = compare_all_solvers_general(
                domain_type=args.domain,
                domain_params=domain_params,
                sources_true=sources_true,
                noise_level=args.noise,
                alpha=args.alpha,
                forward_resolution=fwd_resolution,
                source_resolution=src_resolution,
                quick=args.quick,
                seed=args.seed,
                verbose=True
            )
        
        # Print and save results table
        print_comparison_table(results)
        
        # Save meshes for reproducibility
        try:
            from .mesh import save_meshes
            from .fem_solver import FEMLinearInverseSolver
        except ImportError:
            from mesh import save_meshes
            from fem_solver import FEMLinearInverseSolver
        
        mesh_dir = tracker.output_dir / 'meshes'
        try:
            # Build a representative FEM solver to extract mesh (uses gmsh, same as comparison)
            if args.domain == 'disk':
                fem_for_mesh = FEMLinearInverseSolver(
                    forward_resolution=calibration_params.get('forward_mesh_resolution', 0.1) if calibration_params else 0.1,
                    source_resolution=calibration_params.get('source_grid_resolution', 0.15) if calibration_params else 0.15,
                    verbose=False
                )
            elif args.domain == 'ellipse':
                a = domain_params.get('a', 2.0) if domain_params else 2.0
                b = domain_params.get('b', 1.0) if domain_params else 1.0
                fem_for_mesh = FEMLinearInverseSolver.from_ellipse(
                    a, b,
                    forward_resolution=calibration_params.get('forward_mesh_resolution', 0.1) if calibration_params else 0.1,
                    source_resolution=calibration_params.get('source_grid_resolution', 0.15) if calibration_params else 0.15,
                    verbose=False
                )
            else:
                # Star, square, polygon, brain use polygon mesh
                if args.domain == 'square':
                    vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
                elif args.domain == 'polygon':
                    vertices = domain_params.get('vertices', [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]) if domain_params else [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
                elif args.domain == 'star':
                    n_petals = domain_params.get('n_petals', 5) if domain_params else 5
                    amplitude = domain_params.get('amplitude', 0.3) if domain_params else 0.3
                    n_v = 100  # Match comparison.py which uses 100 vertices
                    theta_v = np.linspace(0, 2*np.pi, n_v, endpoint=False)
                    r_v = 1.0 + amplitude * np.cos(n_petals * theta_v)
                    vertices = [(r_v[i] * np.cos(theta_v[i]), r_v[i] * np.sin(theta_v[i])) for i in range(n_v)]
                else:  # brain
                    try:
                        from .mesh import get_brain_boundary
                    except ImportError:
                        from mesh import get_brain_boundary
                    boundary = get_brain_boundary(n_points=100)  # Match comparison.py
                    vertices = [tuple(p) for p in boundary]
                
                fem_for_mesh = FEMLinearInverseSolver.from_polygon(
                    vertices,
                    forward_resolution=calibration_params.get('forward_mesh_resolution', 0.1) if calibration_params else 0.1,
                    source_resolution=calibration_params.get('source_grid_resolution', 0.15) if calibration_params else 0.15,
                    verbose=False
                )
            
            # Save meshes
            forward_res = calibration_params.get('forward_mesh_resolution', 0.1) if calibration_params else 0.1
            source_res = calibration_params.get('source_grid_resolution', 0.15) if calibration_params else 0.15
            
            saved_mesh_files = save_meshes(
                output_dir=str(mesh_dir),
                forward_mesh=fem_for_mesh.get_mesh_data(),
                source_grid=fem_for_mesh.get_source_grid(),
                domain_type=args.domain,
                forward_resolution=forward_res,
                source_resolution=source_res
            )
            
            # Log mesh artifacts
            for name, filepath in saved_mesh_files.items():
                tracker.artifacts.append({
                    'type': 'mesh',
                    'filename': os.path.basename(filepath),
                    'filepath': filepath,
                    'description': f'{name} for {args.domain}'
                })
            
            print(f"\nMeshes saved to: {mesh_dir}")
            
        except Exception as e:
            print(f"\nNote: Could not save meshes: {e}")
        
        # Save detailed results
        results_data = []
        for r in results:
            result_dict = {
                'solver_name': r.solver_name,
                'method_type': r.method_type,
                'forward_type': r.forward_type,
                'position_rmse': r.position_rmse,
                'intensity_rmse': r.intensity_rmse,
                'boundary_residual': r.boundary_residual,
                'time_seconds': r.time_seconds,
                'iterations': r.iterations,
                'sources_recovered': r.sources_recovered,
            }
            results_data.append(result_dict)
            
            # Log best result metrics
            if r.position_rmse < tracker.metrics.position_rmse or tracker.metrics.position_rmse == 0:
                tracker.log_metrics(
                    position_rmse=r.position_rmse,
                    intensity_rmse=r.intensity_rmse,
                    boundary_residual=r.boundary_residual,
                    time_seconds=r.time_seconds,
                    n_sources_recovered=r.sources_recovered
                )
        
        # Save results JSON
        tracker.save_data(results_data, 'results.json', description='Comparison results')
        
        # Generate and save plot (no GUI)
        if not args.no_plot:
            fig = plot_comparison(results, sources_true, save_path=None,
                                  domain_type=args.domain, domain_params=domain_params)
            tracker.save_figure(fig, 'comparison.png', description='Solver comparison plot')
            print(f"\nFigure saved to: {tracker.output_dir / 'figures' / 'comparison.png'}")
        
        print(f"\nExperiment complete: {tracker.experiment_id}")
        print(f"Database: {tracker.db_path}")


def run_convergence(args):
    """Run mesh convergence study."""
    try:
        from .mesh_convergence import (
            run_forward_mesh_convergence,
            run_inverse_source_grid_convergence,
            run_full_convergence_study
        )
    except ImportError:
        from mesh_convergence import (
            run_forward_mesh_convergence,
            run_inverse_source_grid_convergence,
            run_full_convergence_study
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.forward_only:
        study = run_forward_mesh_convergence(
            domain_type=args.domain,
            verbose=True
        )
        study.plot(
            save_path=str(output_dir / f'{args.domain}_forward_convergence.png'),
            show=True
        )
        print(f"\nOptimal forward resolution: {study.optimal_forward_resolution}")
        
    elif args.inverse_only:
        study = run_inverse_source_grid_convergence(
            domain_type=args.domain,
            verbose=True
        )
        study.plot(
            save_path=str(output_dir / f'{args.domain}_inverse_convergence.png'),
            show=True
        )
        print(f"\nOptimal source grid resolution: {study.optimal_source_resolution}")
        
    else:
        forward_study, inverse_study = run_full_convergence_study(
            domain_type=args.domain,
            output_dir=str(output_dir),
            verbose=True
        )
        print(f"\nOptimal forward resolution: {forward_study.optimal_forward_resolution}")
        print(f"Optimal source grid resolution: {inverse_study.optimal_source_resolution}")


def run_calibrate(args):
    """Run parameter calibration for all domains."""
    try:
        from .calibration import calibrate_all_domains, plot_calibration_results
    except ImportError:
        from calibration import calibrate_all_domains, plot_calibration_results
    
    # calibrate_all_domains saves to stable path + archive
    config = calibrate_all_domains(
        domains=args.domains,
        output_dir=args.output_dir,
        noise_level=args.noise,
        seed=args.seed,
        verbose=True
    )
    
    if args.plot:
        # Use the stable config path
        config_path = os.path.join(args.output_dir, 'calibration_config.json')
        if os.path.exists(config_path):
            # Find the archive directory for plots
            archive_base = os.path.join(args.output_dir, 'archive')
            if os.path.exists(archive_base):
                subdirs = sorted([d for d in os.listdir(archive_base) 
                                 if d.startswith('calibration_')])
                if subdirs:
                    latest_archive = os.path.join(archive_base, subdirs[-1])
                    plot_calibration_results(config_path, latest_archive)
                    print(f"\nPlots saved to: {latest_archive}")


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
    elif args.command == 'calibrate':
        run_calibrate(args)
    elif args.command == 'convergence':
        run_convergence(args)
    elif args.command == 'info':
        run_info(args)


if __name__ == '__main__':
    main()
