"""
Inverse Solver for Source Localization
=======================================
Configurable via JSON file with many optimizer options.

Usage:
    python inverse_solver.py                         # Use default config.json
    python inverse_solver.py --config my.json        # Custom config
    python inverse_solver.py --optimizer L-BFGS-B    # Override optimizer
    python inverse_solver.py --no-live               # Disable live plotting
    python inverse_solver.py --list-optimizers       # Show available optimizers
"""

import numpy as np
import json
import argparse
from pathlib import Path
from scipy.optimize import (
    minimize, differential_evolution, basinhopping, 
    dual_annealing, shgo
)
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from forward_solver import (
    create_disk_mesh,
    solve_poisson_zero_neumann,
    get_boundary_values,
)


# ============================================================================
# AVAILABLE OPTIMIZERS
# ============================================================================

GLOBAL_OPTIMIZERS = [
    'differential_evolution',
    'basinhopping', 
    'dual_annealing',
    'shgo'
]

LOCAL_GRADIENT_OPTIMIZERS = [
    'L-BFGS-B',      # Quasi-Newton, bounded
    'CG',            # Conjugate Gradient
    'BFGS',          # Quasi-Newton
    'Newton-CG',     # Truncated Newton
    'SLSQP',         # Sequential Least Squares
    'trust-constr'   # Trust-region constrained
]

LOCAL_DERIVATIVE_FREE_OPTIMIZERS = [
    'Nelder-Mead',   # Simplex method
    'Powell',        # Direction set method
    'COBYLA'         # Constrained optimization by linear approximation
]

ALL_OPTIMIZERS = GLOBAL_OPTIMIZERS + LOCAL_GRADIENT_OPTIMIZERS + LOCAL_DERIVATIVE_FREE_OPTIMIZERS


def list_optimizers():
    """Print available optimizers with descriptions."""
    print("\n" + "="*70)
    print("AVAILABLE OPTIMIZERS")
    print("="*70)
    
    print("\n🌍 GLOBAL OPTIMIZERS (avoid local minima, slower)")
    print("-" * 50)
    descriptions = {
        'differential_evolution': 'Evolutionary algorithm, very robust',
        'basinhopping': 'Global + local refinement, good for rough landscapes',
        'dual_annealing': 'Simulated annealing variant, handles many local minima',
        'shgo': 'Simplicial homology, finds all local minima'
    }
    for opt in GLOBAL_OPTIMIZERS:
        print(f"  {opt:25s} - {descriptions.get(opt, '')}")
    
    print("\n📈 LOCAL GRADIENT-BASED (fast, may find local minima)")
    print("-" * 50)
    descriptions = {
        'L-BFGS-B': 'Quasi-Newton with bounds, very efficient',
        'CG': 'Conjugate gradient, good for large problems',
        'BFGS': 'Quasi-Newton, standard choice',
        'Newton-CG': 'Truncated Newton, uses Hessian',
        'SLSQP': 'Sequential least squares, handles constraints',
        'trust-constr': 'Trust region, most flexible constraints'
    }
    for opt in LOCAL_GRADIENT_OPTIMIZERS:
        print(f"  {opt:25s} - {descriptions.get(opt, '')}")
    
    print("\n🎲 LOCAL DERIVATIVE-FREE (no gradients needed)")
    print("-" * 50)
    descriptions = {
        'Nelder-Mead': 'Simplex method, robust but slow',
        'Powell': 'Direction set, good without gradients',
        'COBYLA': 'Linear approximation, handles constraints'
    }
    for opt in LOCAL_DERIVATIVE_FREE_OPTIMIZERS:
        print(f"  {opt:25s} - {descriptions.get(opt, '')}")
    
    print("\n" + "="*70)
    print("USAGE TIPS:")
    print("  - Start with 'differential_evolution' for robustness")
    print("  - Use 'L-BFGS-B' with good initial guess for speed")
    print("  - Try 'basinhopping' for balance of global/local search")
    print("="*70 + "\n")


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class Config:
    """Load and manage configuration from JSON file."""
    
    def __init__(self, config_path=None):
        self.config = {}
        
        if config_path and Path(config_path).exists():
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            print("Using default configuration")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default configuration."""
        self.config = {
            "problem": {"n_sources": 4, "radius": 1.0, "noise_level": 0.001},
            "mesh": {"resolution_forward": 0.05, "resolution_inverse_linear": 0.08},
            "sources_true": [
                {"x": -0.3, "y": 0.4, "intensity": 1.0},
                {"x": 0.5, "y": 0.3, "intensity": 1.0},
                {"x": -0.4, "y": -0.4, "intensity": -1.0},
                {"x": 0.3, "y": -0.5, "intensity": -1.0}
            ],
            "solver": {"method": "nonlinear", "run_both": True},
            "nonlinear": {
                "optimizer": "differential_evolution",
                "initial_guess": {"strategy": "circular"},
                "differential_evolution": {"maxiter": 50, "popsize": 10, "seed": 42}
            },
            "linear": {"regularization": "l1", "alpha": 1e-4},
            "visualization": {"live_plot": True, "update_interval": 5, "output_dir": "results"}
        }
    
    def get(self, *keys, default=None):
        """Get nested config value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, value, *keys):
        """Set nested config value."""
        d = self.config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    
    def get_sources_true(self):
        """Convert sources_true to list of tuples."""
        sources_config = self.get('sources_true', default=[])
        return [((s['x'], s['y']), s['intensity']) for s in sources_config]
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Sources: {self.get('problem', 'n_sources')}, "
              f"radius={self.get('problem', 'radius')}, "
              f"noise={self.get('problem', 'noise_level')}")
        print(f"Solver: {self.get('solver', 'method')}, "
              f"run_both={self.get('solver', 'run_both')}")
        
        opt = self.get('nonlinear', 'optimizer')
        opt_type = ("GLOBAL" if opt in GLOBAL_OPTIMIZERS else 
                    "LOCAL-GRADIENT" if opt in LOCAL_GRADIENT_OPTIMIZERS else 
                    "LOCAL-DERIVATIVE-FREE")
        print(f"Optimizer: {opt} ({opt_type})")
        print(f"Initial guess: {self.get('nonlinear', 'initial_guess', 'strategy')}")
        print(f"Live plot: {self.get('visualization', 'live_plot')}")
        print("="*60 + "\n")


# ============================================================================
# LIVE VISUALIZATION
# ============================================================================

class LivePlotter:
    """Live visualization during optimization."""
    
    def __init__(self, mesh, sources_true=None, update_interval=5):
        self.mesh = mesh
        self.sources_true = sources_true
        self.update_interval = update_interval
        self.iteration = 0
        self.history = []
        
        coords = mesh.geometry.x
        mesh.topology.create_connectivity(2, 0)
        cells = mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        self.tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        self.coords = coords
        
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Inverse Problem - Live Optimization')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    def update(self, sources, misfit, angles_measured=None, values_measured=None):
        """Update visualization."""
        self.iteration += 1
        self.history.append(misfit)
        
        if self.iteration % self.update_interval != 0:
            return
        
        for ax in self.axes:
            ax.clear()
        
        self.axes[0].set_title(f'Solution (iter {self.iteration})')
        self.axes[0].set_aspect('equal')
        self.axes[1].set_title('Boundary Fit')
        self.axes[1].grid(True, alpha=0.3)
        self.axes[2].set_title(f'Convergence (misfit={misfit:.2e})')
        self.axes[2].set_yscale('log')
        self.axes[2].grid(True, alpha=0.3)
        
        try:
            u = solve_poisson_zero_neumann(self.mesh, sources)
            u_vals = u.x.array[:len(self.coords)] if len(u.x.array) != len(self.coords) else u.x.array
            
            self.axes[0].triplot(self.tri, 'k-', linewidth=0.2, alpha=0.2)
            self.axes[0].tricontourf(self.tri, u_vals, levels=30, cmap='viridis', alpha=0.8)
            
            if self.sources_true:
                for (x, y), q in self.sources_true:
                    color = 'green' if q > 0 else 'purple'
                    self.axes[0].plot(x, y, 'o', color=color, markersize=18,
                                     markerfacecolor='none', markeredgewidth=3)
            
            for (x, y), q in sources:
                color = 'red' if q > 0 else 'blue'
                marker = '+' if q > 0 else '*'
                self.axes[0].plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
            
            angles_comp, values_comp = get_boundary_values(u)
            if angles_measured is not None:
                self.axes[1].plot(angles_measured, values_measured, 'b-', linewidth=2, label='Measured')
            self.axes[1].plot(angles_comp, values_comp, 'r--', linewidth=2, label='Current')
            self.axes[1].legend()
            
        except Exception as e:
            self.axes[0].text(0, 0, f'Error: {e}', ha='center')
        
        if len(self.history) > 1:
            self.axes[2].plot(range(1, len(self.history) + 1), self.history, 'b.-')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def finish(self, save_path=None):
        plt.ioff()
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# NONLINEAR INVERSE SOLVER
# ============================================================================

class NonlinearInverseSolver:
    """Nonlinear inverse solver with multiple optimizer options."""
    
    def __init__(self, config: Config):
        self.config = config
        self.n_sources = config.get('problem', 'n_sources')
        self.radius = config.get('problem', 'radius')
        
        resolution = config.get('mesh', 'resolution_forward')
        print(f"Creating mesh (resolution={resolution})...")
        self.mesh, _, _ = create_disk_mesh(self.radius, resolution)
        
        self.angles_measured = None
        self.values_measured = None
        self.interpolator = None
        self.history = []
        self.live_plotter = None
        self.live_plot = False
        self.eval_count = 0
        
    def set_measured_data(self, angles, values):
        """Set measured boundary data."""
        sort_idx = np.argsort(angles)
        self.angles_measured = angles[sort_idx]
        self.values_measured = values[sort_idx]
        
        angles_ext = np.concatenate([
            self.angles_measured - 2*np.pi,
            self.angles_measured,
            self.angles_measured + 2*np.pi
        ])
        values_ext = np.tile(self.values_measured, 3)
        self.interpolator = interp1d(angles_ext, values_ext, kind='linear')
        
    def params_to_sources(self, params):
        """Convert flat parameters to sources list."""
        sources = []
        n = self.n_sources
        
        for i in range(n - 1):
            x, y, q = params[3*i], params[3*i + 1], params[3*i + 2]
            sources.append(((x, y), q))
        
        x_last, y_last = params[3*(n-1)], params[3*(n-1) + 1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def sources_to_params(self, sources):
        """Convert sources to flat parameters."""
        params = []
        for i, ((x, y), q) in enumerate(sources):
            params.extend([x, y])
            if i < len(sources) - 1:
                params.append(q)
        return np.array(params)
    
    def get_initial_guess(self):
        """Generate initial guess based on config strategy."""
        strategy = self.config.get('nonlinear', 'initial_guess', 'strategy', default='circular')
        n = self.n_sources
        r = 0.5 * self.radius
        
        if strategy == 'circular':
            # Place sources evenly on a circle
            params = []
            for i in range(n):
                angle = 2 * np.pi * i / n
                params.extend([r * np.cos(angle), r * np.sin(angle)])
                if i < n - 1:
                    params.append(1.0 if i % 2 == 0 else -1.0)
            return np.array(params)
        
        elif strategy == 'random':
            seed = self.config.get('nonlinear', 'initial_guess', 'random_seed', default=42)
            np.random.seed(seed)
            params = []
            for i in range(n):
                # Random point inside disk
                while True:
                    x, y = np.random.uniform(-r, r, 2)
                    if x**2 + y**2 < r**2:
                        break
                params.extend([x, y])
                if i < n - 1:
                    params.append(np.random.choice([-1.0, 1.0]))
            return np.array(params)
        
        elif strategy == 'grid':
            # Grid pattern
            grid_size = int(np.ceil(np.sqrt(n)))
            params = []
            idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    if idx >= n:
                        break
                    x = -r + 2*r * (i + 0.5) / grid_size
                    y = -r + 2*r * (j + 0.5) / grid_size
                    params.extend([x, y])
                    if idx < n - 1:
                        params.append(1.0 if idx % 2 == 0 else -1.0)
                    idx += 1
            return np.array(params)
        
        elif strategy == 'custom':
            custom = self.config.get('nonlinear', 'initial_guess', 'custom_sources', default=[])
            sources = [((s['x'], s['y']), s['intensity']) for s in custom]
            return self.sources_to_params(sources)
        
        else:
            raise ValueError(f"Unknown initial guess strategy: {strategy}")
    
    def objective(self, params):
        """Objective function."""
        self.eval_count += 1
        sources = self.params_to_sources(params)
        
        # Penalty for sources outside domain
        for (x, y), q in sources:
            if x**2 + y**2 >= (0.95 * self.radius)**2:
                return 1e10
        
        try:
            u = solve_poisson_zero_neumann(self.mesh, sources)
            angles_comp, values_comp = get_boundary_values(u)
            values_measured_interp = self.interpolator(angles_comp)
            misfit = np.sum((values_comp - values_measured_interp)**2)
            
            self.history.append({'misfit': misfit, 'sources': sources, 'eval': self.eval_count})
            
            if self.live_plot and self.live_plotter:
                self.live_plotter.update(sources, misfit, self.angles_measured, self.values_measured)
            
            if len(self.history) % 20 == 0:
                print(f"  Eval {self.eval_count}: misfit = {misfit:.6e}")
            
            return misfit
        except:
            return 1e10
    
    def solve(self, sources_true=None):
        """Solve using configured optimizer."""
        self.history = []
        self.eval_count = 0
        
        optimizer = self.config.get('nonlinear', 'optimizer')
        opt_params = self.config.get('nonlinear', optimizer, default={})
        # Remove comment keys
        opt_params = {k: v for k, v in opt_params.items() if not k.startswith('_')}
        
        self.live_plot = self.config.get('visualization', 'live_plot')
        update_interval = self.config.get('visualization', 'update_interval')
        
        if self.live_plot:
            self.live_plotter = LivePlotter(
                self.mesh, sources_true=sources_true, update_interval=update_interval
            )
        
        # Setup bounds
        n = self.n_sources
        r_max = 0.9 * self.radius
        bounds = []
        for i in range(n):
            bounds.append((-r_max, r_max))  # x
            bounds.append((-r_max, r_max))  # y
            if i < n - 1:
                bounds.append((-5.0, 5.0))  # q
        
        x0 = self.get_initial_guess()
        
        opt_type = ("GLOBAL" if optimizer in GLOBAL_OPTIMIZERS else 
                    "LOCAL-GRADIENT" if optimizer in LOCAL_GRADIENT_OPTIMIZERS else 
                    "LOCAL-DERIVATIVE-FREE")
        print(f"\n{'='*60}")
        print(f"Running: {optimizer} ({opt_type})")
        print(f"{'='*60}")
        print(f"Parameters: {opt_params}")
        print(f"Initial guess strategy: {self.config.get('nonlinear', 'initial_guess', 'strategy')}")
        
        # ============ GLOBAL OPTIMIZERS ============
        if optimizer == 'differential_evolution':
            result = differential_evolution(
                self.objective, bounds,
                maxiter=opt_params.get('maxiter', 100),
                popsize=opt_params.get('popsize', 15),
                tol=opt_params.get('tol', 1e-7),
                mutation=opt_params.get('mutation', (0.5, 1)),
                recombination=opt_params.get('recombination', 0.7),
                seed=opt_params.get('seed', None),
                polish=opt_params.get('polish', True),
                strategy=opt_params.get('strategy', 'best1bin'),
                workers=1, updating='deferred', disp=True
            )
            
        elif optimizer == 'basinhopping':
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": bounds
            }
            result = basinhopping(
                self.objective, x0,
                niter=opt_params.get('niter', 100),
                T=opt_params.get('T', 1.0),
                stepsize=opt_params.get('stepsize', 0.3),
                seed=opt_params.get('seed', None),
                interval=opt_params.get('interval', 50),
                niter_success=opt_params.get('niter_success', None),
                minimizer_kwargs=minimizer_kwargs
            )
            
        elif optimizer == 'dual_annealing':
            result = dual_annealing(
                self.objective, bounds,
                maxiter=opt_params.get('maxiter', 1000),
                initial_temp=opt_params.get('initial_temp', 5230.0),
                restart_temp_ratio=opt_params.get('restart_temp_ratio', 2e-5),
                visit=opt_params.get('visit', 2.62),
                accept=opt_params.get('accept', -5.0),
                seed=opt_params.get('seed', None),
                no_local_search=opt_params.get('no_local_search', False)
            )
            
        elif optimizer == 'shgo':
            result = shgo(
                self.objective, bounds,
                n=opt_params.get('n', 100),
                iters=opt_params.get('iters', 1),
                sampling_method=opt_params.get('sampling_method', 'simplicial')
            )
        
        # ============ LOCAL GRADIENT-BASED ============
        elif optimizer == 'L-BFGS-B':
            result = minimize(
                self.objective, x0, method='L-BFGS-B', bounds=bounds,
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'maxfun': opt_params.get('maxfun', 15000),
                    'ftol': opt_params.get('ftol', 1e-9),
                    'gtol': opt_params.get('gtol', 1e-7),
                    'eps': opt_params.get('eps', 1e-8),
                    'maxcor': opt_params.get('maxcor', 10),
                    'disp': True
                }
            )
            
        elif optimizer == 'CG':
            result = minimize(
                self.objective, x0, method='CG',
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'gtol': opt_params.get('gtol', 1e-7),
                    'eps': opt_params.get('eps', 1e-8),
                    'disp': True
                }
            )
            
        elif optimizer == 'BFGS':
            result = minimize(
                self.objective, x0, method='BFGS',
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'gtol': opt_params.get('gtol', 1e-7),
                    'eps': opt_params.get('eps', 1e-8),
                    'disp': True
                }
            )
            
        elif optimizer == 'Newton-CG':
            result = minimize(
                self.objective, x0, method='Newton-CG',
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'xtol': opt_params.get('xtol', 1e-8),
                    'eps': opt_params.get('eps', 1e-8),
                    'disp': True
                }
            )
            
        elif optimizer == 'SLSQP':
            result = minimize(
                self.objective, x0, method='SLSQP', bounds=bounds,
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'ftol': opt_params.get('ftol', 1e-9),
                    'eps': opt_params.get('eps', 1e-8),
                    'disp': True
                }
            )
            
        elif optimizer == 'trust-constr':
            result = minimize(
                self.objective, x0, method='trust-constr', bounds=bounds,
                options={
                    'maxiter': opt_params.get('maxiter', 200),
                    'gtol': opt_params.get('gtol', 1e-8),
                    'xtol': opt_params.get('xtol', 1e-8),
                    'initial_tr_radius': opt_params.get('initial_tr_radius', 1.0),
                    'max_tr_radius': opt_params.get('max_tr_radius', 1000.0),
                    'verbose': 2
                }
            )
        
        # ============ LOCAL DERIVATIVE-FREE ============
        elif optimizer == 'Nelder-Mead':
            result = minimize(
                self.objective, x0, method='Nelder-Mead',
                options={
                    'maxiter': opt_params.get('maxiter', 500),
                    'maxfev': opt_params.get('maxfev', 10000),
                    'xatol': opt_params.get('xatol', 1e-8),
                    'fatol': opt_params.get('fatol', 1e-8),
                    'adaptive': opt_params.get('adaptive', True),
                    'disp': True
                }
            )
            
        elif optimizer == 'Powell':
            result = minimize(
                self.objective, x0, method='Powell',
                options={
                    'maxiter': opt_params.get('maxiter', 500),
                    'maxfev': opt_params.get('maxfev', 10000),
                    'ftol': opt_params.get('ftol', 1e-8),
                    'xtol': opt_params.get('xtol', 1e-8),
                    'disp': True
                }
            )
            
        elif optimizer == 'COBYLA':
            # COBYLA doesn't use bounds directly, need constraints
            constraints = []
            for i, (lb, ub) in enumerate(bounds):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, lb=lb: x[i] - lb})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i, ub=ub: ub - x[i]})
            
            result = minimize(
                self.objective, x0, method='COBYLA',
                constraints=constraints,
                options={
                    'maxiter': opt_params.get('maxiter', 1000),
                    'rhobeg': opt_params.get('rhobeg', 0.5),
                    'tol': opt_params.get('tol', 1e-8),
                    'disp': True
                }
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}. Use --list-optimizers to see options.")
        
        sources_recovered = self.params_to_sources(result.x)
        
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Final misfit: {result.fun:.6e}")
        print(f"Function evaluations: {self.eval_count}")
        print(f"Success: {result.success if hasattr(result, 'success') else 'N/A'}")
        
        # Finalize
        if self.live_plot and self.live_plotter:
            output_dir = self.config.get('visualization', 'output_dir')
            self.live_plotter.finish(save_path=f"{output_dir}/inverse_live_final.png")
        
        return sources_recovered, result
    
    def compare_results(self, sources_true, sources_recovered, save_path=None):
        """Compare true and recovered sources."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        # True
        ax = axes[0]
        u_true = solve_poisson_zero_neumann(self.mesh, sources_true)
        u_vals = u_true.x.array[:len(coords)] if len(u_true.x.array) != len(coords) else u_true.x.array
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        tcf = ax.tricontourf(tri, u_vals, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(tcf, ax=ax, label='u')
        for (x, y), q in sources_true:
            color, marker = ('red', '+') if q > 0 else ('blue', '*')
            ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
        ax.set_title('True Sources')
        ax.set_aspect('equal')
        
        # Recovered
        ax = axes[1]
        u_rec = solve_poisson_zero_neumann(self.mesh, sources_recovered)
        u_vals = u_rec.x.array[:len(coords)] if len(u_rec.x.array) != len(coords) else u_rec.x.array
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        tcf = ax.tricontourf(tri, u_vals, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(tcf, ax=ax, label='u')
        for (x, y), q in sources_recovered:
            color, marker = ('red', '+') if q > 0 else ('blue', '*')
            ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
        ax.set_title('Recovered Sources')
        ax.set_aspect('equal')
        
        # Boundary
        ax = axes[2]
        angles_true, values_true = get_boundary_values(u_true)
        angles_rec, values_rec = get_boundary_values(u_rec)
        ax.plot(angles_true, values_true, 'b-', linewidth=2, label='True')
        ax.plot(angles_rec, values_rec, 'r--', linewidth=2, label='Recovered')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Boundary Value')
        ax.set_title('Boundary Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print comparison
        print("\n" + "="*60)
        print("TRUE vs RECOVERED")
        print("="*60)
        print("\nTrue:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
        print("\nRecovered:")
        for i, ((x, y), q) in enumerate(sources_recovered):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
        
        # Compute error
        print("\nPosition errors:")
        for i, ((xr, yr), qr) in enumerate(sources_recovered):
            min_dist = min(np.sqrt((xr-xt)**2 + (yr-yt)**2) 
                          for (xt, yt), _ in sources_true)
            print(f"  Source {i+1}: {min_dist:.4f}")
    
    def plot_convergence(self, save_path=None):
        """Plot convergence history."""
        if not self.history:
            print("No history to plot")
            return
        
        misfits = [h['misfit'] for h in self.history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(range(1, len(misfits) + 1), misfits, 'b.-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Misfit (log scale)')
        ax.set_title(f'Convergence History - {self.config.get("nonlinear", "optimizer")}')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# LINEAR INVERSE SOLVER (kept simple for brevity)
# ============================================================================

class LinearInverseSolver:
    """Linear inverse solver - sources at mesh nodes."""
    
    def __init__(self, config: Config):
        self.config = config
        self.radius = config.get('problem', 'radius')
        resolution = config.get('mesh', 'resolution_inverse_linear')
        
        print(f"Creating mesh (resolution={resolution})...")
        self.mesh, _, _ = create_disk_mesh(self.radius, resolution)
        
        self.all_coords = self.mesh.geometry.x[:, :2]
        radii = np.sqrt(self.all_coords[:, 0]**2 + self.all_coords[:, 1]**2)
        
        self.interior_mask = radii < 0.95 * self.radius
        self.interior_coords = self.all_coords[self.interior_mask]
        self.n_interior = len(self.interior_coords)
        
        self.boundary_mask = radii > 0.9 * self.radius
        self.boundary_coords = self.all_coords[self.boundary_mask]
        self.n_boundary = len(self.boundary_coords)
        self.boundary_angles = np.arctan2(self.boundary_coords[:, 1], self.boundary_coords[:, 0])
        
        sort_idx = np.argsort(self.boundary_angles)
        self.boundary_angles = self.boundary_angles[sort_idx]
        
        print(f"Interior: {self.n_interior}, Boundary: {self.n_boundary}")
        
        self.G = None
        
    def build_greens_matrix(self):
        """Build Green's matrix."""
        print(f"\nBuilding Green's matrix ({self.n_boundary} x {self.n_interior})...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j in range(self.n_interior):
            if j % 50 == 0:
                print(f"  Column {j}/{self.n_interior}...")
            
            x_j, y_j = self.interior_coords[j]
            sources = [((x_j, y_j), 1.0), ((0, 0), -1.0)]
            
            u = solve_poisson_zero_neumann(self.mesh, sources)
            angles_computed, values_computed = get_boundary_values(u)
            
            interp = interp1d(
                np.concatenate([angles_computed - 2*np.pi, angles_computed, angles_computed + 2*np.pi]),
                np.tile(values_computed, 3), kind='linear'
            )
            self.G[:, j] = interp(self.boundary_angles)
        
        print("Green's matrix built!")
    
    def solve(self, u_measured, sources_true=None):
        """Solve linear inverse problem."""
        if self.G is None:
            self.build_greens_matrix()
        
        reg = self.config.get('linear', 'regularization')
        alpha = self.config.get('linear', 'alpha')
        
        print(f"\nSolving (regularization={reg}, alpha={alpha})...")
        
        if reg == 'l2':
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q = np.linalg.solve(GtG + alpha * np.eye(self.n_interior), Gtu)
        elif reg == 'l1':
            q = self._solve_l1(u_measured, alpha)
        else:
            raise ValueError(f"Unknown regularization: {reg}")
        
        q = q - np.mean(q)
        
        threshold = 0.1 * np.max(np.abs(q))
        sources = [((self.interior_coords[j, 0], self.interior_coords[j, 1]), q[j]) 
                   for j in range(self.n_interior) if np.abs(q[j]) > threshold]
        
        print(f"Found {len(sources)} significant sources")
        return q, sources
    
    def _solve_l1(self, u_measured, alpha):
        """L1 via IRLS."""
        max_iter = self.config.get('linear', 'l1_max_iter', default=50)
        q = np.zeros(self.n_interior)
        epsilon = 1e-4
        
        for iteration in range(max_iter):
            W = np.diag(1.0 / (np.abs(q) + epsilon))
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            
            if np.linalg.norm(q_new - q) < 1e-6:
                print(f"  Converged at iteration {iteration}")
                break
            q = q_new
        return q
    
    def plot_results(self, q, sources_true=None, save_path=None):
        """Plot results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        ax = axes[0]
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)
        scatter = ax.scatter(
            self.interior_coords[:, 0], self.interior_coords[:, 1],
            c=q, cmap='RdBu_r', s=30,
            vmin=-np.max(np.abs(q)), vmax=np.max(np.abs(q))
        )
        plt.colorbar(scatter, ax=ax, label='q')
        if sources_true:
            for (x, y), intensity in sources_true:
                marker = 'o' if intensity > 0 else 's'
                ax.plot(x, y, marker, color='black', markersize=15,
                       markerfacecolor='none', markeredgewidth=3)
        ax.set_title('Recovered (Linear)')
        ax.set_aspect('equal')
        
        ax = axes[1]
        if hasattr(self, 'u_measured'):
            ax.plot(self.boundary_angles, self.u_measured, 'b-', linewidth=2, label='Measured')
        ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Reconstructed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Boundary Fit')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# HELPER
# ============================================================================

def generate_synthetic_data(sources, noise_level=0.0, mesh_resolution=0.05):
    """Generate synthetic boundary measurements."""
    mesh, _, _ = create_disk_mesh(resolution=mesh_resolution)
    u = solve_poisson_zero_neumann(mesh, sources)
    angles, values = get_boundary_values(u)
    if noise_level > 0:
        values = values + np.random.normal(0, noise_level, len(values))
    return angles, values


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inverse Source Localization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inverse_solver.py                              # Default config
  python inverse_solver.py --optimizer L-BFGS-B         # Use L-BFGS-B
  python inverse_solver.py --optimizer basinhopping     # Use basinhopping
  python inverse_solver.py --no-live                    # Disable live plot
  python inverse_solver.py --list-optimizers            # Show all optimizers
        """
    )
    parser.add_argument('--config', type=str, default='src/config.json',
                        help='Path to config JSON file')
    parser.add_argument('--no-live', action='store_true',
                        help='Disable live plotting')
    parser.add_argument('--optimizer', type=str, choices=ALL_OPTIMIZERS,
                        help='Override optimizer')
    parser.add_argument('--method', type=str, choices=['nonlinear', 'linear', 'both'],
                        help='Solver method')
    parser.add_argument('--list-optimizers', action='store_true',
                        help='List available optimizers and exit')
    args = parser.parse_args()
    
    if args.list_optimizers:
        list_optimizers()
        return
    
    # Load config
    config = Config(args.config)
    
    # Command-line overrides
    if args.no_live:
        config.set(False, 'visualization', 'live_plot')
    if args.optimizer:
        config.set(args.optimizer, 'nonlinear', 'optimizer')
    if args.method:
        if args.method == 'both':
            config.set(True, 'solver', 'run_both')
        else:
            config.set(args.method, 'solver', 'method')
            config.set(False, 'solver', 'run_both')
    
    config.print_summary()
    
    # Generate data
    sources_true = config.get_sources_true()
    print("True sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.2f}, {y:+.2f}), q={q:+.1f}")
    
    noise = config.get('problem', 'noise_level')
    resolution = config.get('mesh', 'resolution_forward')
    print(f"\nGenerating data (noise={noise})...")
    angles, values = generate_synthetic_data(sources_true, noise, resolution)
    
    output_dir = config.get('visualization', 'output_dir')
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run solvers
    run_both = config.get('solver', 'run_both')
    method = config.get('solver', 'method')
    
    if run_both or method == 'nonlinear':
        print("\n" + "="*60)
        print("NONLINEAR SOLVER")
        print("="*60)
        
        solver = NonlinearInverseSolver(config)
        solver.set_measured_data(angles, values)
        sources_rec, _ = solver.solve(sources_true=sources_true)
        solver.compare_results(sources_true, sources_rec, 
                              save_path=f"{output_dir}/inverse_nonlinear.png")
        if config.get('visualization', 'plot_convergence'):
            solver.plot_convergence(save_path=f"{output_dir}/convergence.png")
    
    if run_both or method == 'linear':
        print("\n" + "="*60)
        print("LINEAR SOLVER")
        print("="*60)
        
        solver = LinearInverseSolver(config)
        solver.build_greens_matrix()
        
        interp = interp1d(
            np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi]),
            np.tile(values, 3), kind='linear'
        )
        u_measured = interp(solver.boundary_angles)
        solver.u_measured = u_measured
        
        q, sources_lin = solver.solve(u_measured, sources_true=sources_true)
        solver.plot_results(q, sources_true, save_path=f"{output_dir}/inverse_linear.png")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
