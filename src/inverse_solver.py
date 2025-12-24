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
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay
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

GLOBAL_OPTIMIZERS = ['differential_evolution', 'basinhopping', 'dual_annealing', 'shgo']
LOCAL_GRADIENT_OPTIMIZERS = ['L-BFGS-B', 'CG', 'BFGS', 'Newton-CG', 'SLSQP', 'trust-constr']
LOCAL_DERIVATIVE_FREE_OPTIMIZERS = ['Nelder-Mead', 'Powell', 'COBYLA']
ALL_OPTIMIZERS = GLOBAL_OPTIMIZERS + LOCAL_GRADIENT_OPTIMIZERS + LOCAL_DERIVATIVE_FREE_OPTIMIZERS

LINEAR_REGULARIZATIONS = ['l1', 'l2', 'tv_admm', 'tv_primal_dual']


def list_optimizers():
    """Print available optimizers."""
    print("\n" + "="*70)
    print("AVAILABLE OPTIMIZERS")
    print("="*70)
    
    print("\n🌍 GLOBAL (avoid local minima, slower)")
    for opt in GLOBAL_OPTIMIZERS:
        print(f"  {opt}")
    
    print("\n📈 LOCAL GRADIENT-BASED (fast, needs good initial guess)")
    for opt in LOCAL_GRADIENT_OPTIMIZERS:
        print(f"  {opt}")
    
    print("\n🎲 LOCAL DERIVATIVE-FREE")
    for opt in LOCAL_DERIVATIVE_FREE_OPTIMIZERS:
        print(f"  {opt}")
    
    print("\n📊 LINEAR REGULARIZATIONS")
    for reg in LINEAR_REGULARIZATIONS:
        print(f"  {reg}")
    print("="*70 + "\n")


# ============================================================================
# CONFIGURATION
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
            "nonlinear": {"optimizer": "differential_evolution"},
            "linear": {"regularization": "l1", "l1": {"alpha": 1e-4}},
            "visualization": {"live_plot": True, "update_interval": 5, "output_dir": "results"}
        }
    
    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, value, *keys):
        d = self.config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value
    
    def get_sources_true(self):
        sources_config = self.get('sources_true', default=[])
        return [((s['x'], s['y']), s['intensity']) for s in sources_config]
    
    def print_summary(self):
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Sources: {self.get('problem', 'n_sources')}")
        print(f"Solver: {self.get('solver', 'method')}")
        print(f"Nonlinear optimizer: {self.get('nonlinear', 'optimizer')}")
        print(f"Linear regularization: {self.get('linear', 'regularization')}")
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
        sources = []
        n = self.n_sources
        
        for i in range(n - 1):
            x, y, q = params[3*i], params[3*i + 1], params[3*i + 2]
            sources.append(((x, y), q))
        
        x_last, y_last = params[3*(n-1)], params[3*(n-1) + 1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def get_initial_guess(self):
        strategy = self.config.get('nonlinear', 'initial_guess', 'strategy', default='circular')
        n = self.n_sources
        r = 0.5 * self.radius
        
        if strategy == 'circular':
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
                while True:
                    x, y = np.random.uniform(-r, r, 2)
                    if x**2 + y**2 < r**2:
                        break
                params.extend([x, y])
                if i < n - 1:
                    params.append(np.random.choice([-1.0, 1.0]))
            return np.array(params)
        
        elif strategy == 'custom':
            custom = self.config.get('nonlinear', 'initial_guess', 'custom_sources', default=[])
            sources = [((s['x'], s['y']), s['intensity']) for s in custom]
            params = []
            for i, ((x, y), q) in enumerate(sources):
                params.extend([x, y])
                if i < len(sources) - 1:
                    params.append(q)
            return np.array(params)
        
        return np.zeros(3*n - 1)
    
    def objective(self, params):
        self.eval_count += 1
        sources = self.params_to_sources(params)
        
        for (x, y), q in sources:
            if x**2 + y**2 >= (0.95 * self.radius)**2:
                return 1e10
        
        try:
            u = solve_poisson_zero_neumann(self.mesh, sources)
            angles_comp, values_comp = get_boundary_values(u)
            values_measured_interp = self.interpolator(angles_comp)
            misfit = np.sum((values_comp - values_measured_interp)**2)
            
            self.history.append({'misfit': misfit, 'sources': sources})
            
            if self.live_plot and self.live_plotter:
                self.live_plotter.update(sources, misfit, self.angles_measured, self.values_measured)
            
            if len(self.history) % 20 == 0:
                print(f"  Eval {self.eval_count}: misfit = {misfit:.6e}")
            
            return misfit
        except:
            return 1e10
    
    def solve(self, sources_true=None):
        self.history = []
        self.eval_count = 0
        
        optimizer = self.config.get('nonlinear', 'optimizer')
        opt_params = self.config.get('nonlinear', optimizer, default={})
        opt_params = {k: v for k, v in opt_params.items() if not k.startswith('_')}
        
        self.live_plot = self.config.get('visualization', 'live_plot')
        update_interval = self.config.get('visualization', 'update_interval')
        
        if self.live_plot:
            self.live_plotter = LivePlotter(
                self.mesh, sources_true=sources_true, update_interval=update_interval
            )
        
        n = self.n_sources
        r_max = 0.9 * self.radius
        bounds = []
        for i in range(n):
            bounds.append((-r_max, r_max))
            bounds.append((-r_max, r_max))
            if i < n - 1:
                bounds.append((-5.0, 5.0))
        
        x0 = self.get_initial_guess()
        
        print(f"\nRunning: {optimizer}")
        
        if optimizer == 'differential_evolution':
            result = differential_evolution(
                self.objective, bounds,
                maxiter=opt_params.get('maxiter', 100),
                popsize=opt_params.get('popsize', 15),
                seed=opt_params.get('seed', None),
                workers=1, updating='deferred', disp=True
            )
        elif optimizer == 'basinhopping':
            result = basinhopping(
                self.objective, x0,
                niter=opt_params.get('niter', 100),
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds}
            )
        elif optimizer == 'dual_annealing':
            result = dual_annealing(self.objective, bounds, maxiter=opt_params.get('maxiter', 1000))
        elif optimizer == 'shgo':
            result = shgo(self.objective, bounds)
        elif optimizer in LOCAL_GRADIENT_OPTIMIZERS:
            result = minimize(self.objective, x0, method=optimizer, bounds=bounds,
                            options={'maxiter': opt_params.get('maxiter', 200), 'disp': True})
        elif optimizer in LOCAL_DERIVATIVE_FREE_OPTIMIZERS:
            result = minimize(self.objective, x0, method=optimizer,
                            options={'maxiter': opt_params.get('maxiter', 500), 'disp': True})
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        sources_recovered = self.params_to_sources(result.x)
        print(f"\nFinal misfit: {result.fun:.6e}")
        
        if self.live_plot and self.live_plotter:
            output_dir = self.config.get('visualization', 'output_dir')
            self.live_plotter.finish(save_path=f"{output_dir}/inverse_live_final.png")
        
        return sources_recovered, result
    
    def compare_results(self, sources_true, sources_recovered, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        for idx, (sources, title) in enumerate([(sources_true, 'True'), (sources_recovered, 'Recovered')]):
            ax = axes[idx]
            u = solve_poisson_zero_neumann(self.mesh, sources)
            u_vals = u.x.array[:len(coords)] if len(u.x.array) != len(coords) else u.x.array
            ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
            tcf = ax.tricontourf(tri, u_vals, levels=50, cmap='viridis', alpha=0.8)
            plt.colorbar(tcf, ax=ax, label='u')
            for (x, y), q in sources:
                color, marker = ('red', '+') if q > 0 else ('blue', '*')
                ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
            ax.set_title(f'{title} Sources')
            ax.set_aspect('equal')
        
        ax = axes[2]
        u_true = solve_poisson_zero_neumann(self.mesh, sources_true)
        u_rec = solve_poisson_zero_neumann(self.mesh, sources_recovered)
        angles_true, values_true = get_boundary_values(u_true)
        angles_rec, values_rec = get_boundary_values(u_rec)
        ax.plot(angles_true, values_true, 'b-', linewidth=2, label='True')
        ax.plot(angles_rec, values_rec, 'r--', linewidth=2, label='Recovered')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Boundary Comparison')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nTrue vs Recovered:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  True {i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.3f}")
        for i, ((x, y), q) in enumerate(sources_recovered):
            print(f"  Rec  {i+1}: ({x:+.3f}, {y:+.3f}), q={q:+.3f}")


# ============================================================================
# LINEAR INVERSE SOLVER (with TV support)
# ============================================================================

class LinearInverseSolver:
    """Linear inverse solver with L1, L2, and TV regularization."""
    
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
        self.D = None  # Gradient operator for TV
        self.edges = None
        
    def build_greens_matrix(self):
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
    
    def build_gradient_operator(self):
        """Build discrete gradient operator D for TV regularization."""
        print("\nBuilding gradient operator for TV...")
        
        tri = Delaunay(self.interior_coords)
        
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges_set.add(edge)
        
        self.edges = list(edges_set)
        n_edges = len(self.edges)
        print(f"Number of edges: {n_edges}")
        
        rows, cols, data = [], [], []
        for e, (i, j) in enumerate(self.edges):
            rows.extend([e, e])
            cols.extend([i, j])
            data.extend([1.0, -1.0])
        
        self.D = csr_matrix((data, (rows, cols)), shape=(n_edges, self.n_interior))
        print("Gradient operator built!")
    
    def soft_threshold(self, x, threshold):
        """Soft thresholding for L1 proximal operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def solve(self, u_measured, sources_true=None):
        if self.G is None:
            self.build_greens_matrix()
        
        reg = self.config.get('linear', 'regularization')
        print(f"\nSolving (regularization={reg})...")
        
        if reg == 'l2':
            alpha = self.config.get('linear', 'l2', 'alpha', default=1e-4)
            q = self._solve_l2(u_measured, alpha)
            
        elif reg == 'l1':
            alpha = self.config.get('linear', 'l1', 'alpha', default=1e-4)
            max_iter = self.config.get('linear', 'l1', 'max_iter', default=50)
            q = self._solve_l1(u_measured, alpha, max_iter)
            
        elif reg == 'tv_admm':
            if self.D is None:
                self.build_gradient_operator()
            alpha = self.config.get('linear', 'tv_admm', 'alpha', default=1e-3)
            rho = self.config.get('linear', 'tv_admm', 'rho', default=1.0)
            max_iter = self.config.get('linear', 'tv_admm', 'max_iter', default=200)
            tol = self.config.get('linear', 'tv_admm', 'tol', default=1e-6)
            q, _ = self._solve_tv_admm(u_measured, alpha, rho, max_iter, tol)
            
        elif reg == 'tv_primal_dual':
            if self.D is None:
                self.build_gradient_operator()
            alpha = self.config.get('linear', 'tv_primal_dual', 'alpha', default=1e-3)
            tau = self.config.get('linear', 'tv_primal_dual', 'tau', default=0.1)
            sigma = self.config.get('linear', 'tv_primal_dual', 'sigma', default=0.1)
            max_iter = self.config.get('linear', 'tv_primal_dual', 'max_iter', default=500)
            tol = self.config.get('linear', 'tv_primal_dual', 'tol', default=1e-6)
            q, _ = self._solve_tv_primal_dual(u_measured, alpha, tau, sigma, max_iter, tol)
            
        else:
            raise ValueError(f"Unknown regularization: {reg}")
        
        q = q - np.mean(q)
        
        threshold = 0.1 * np.max(np.abs(q))
        sources = [((self.interior_coords[j, 0], self.interior_coords[j, 1]), q[j]) 
                   for j in range(self.n_interior) if np.abs(q[j]) > threshold]
        
        print(f"Found {len(sources)} significant sources")
        return q, sources
    
    def _solve_l2(self, u_measured, alpha):
        """L2 (Tikhonov) regularization."""
        print(f"  L2 regularization (alpha={alpha})")
        GtG = self.G.T @ self.G
        Gtu = self.G.T @ u_measured
        return np.linalg.solve(GtG + alpha * np.eye(self.n_interior), Gtu)
    
    def _solve_l1(self, u_measured, alpha, max_iter):
        """L1 via IRLS."""
        print(f"  L1 regularization (alpha={alpha}, max_iter={max_iter})")
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
    
    def _solve_tv_admm(self, u_measured, alpha, rho, max_iter, tol):
        """
        TV regularization via ADMM.
        
        min_q ||Gq - u||^2 + α||Dq||_1
        """
        print(f"  TV-ADMM (alpha={alpha}, rho={rho}, max_iter={max_iter})")
        
        n = self.n_interior
        n_edges = self.D.shape[0]
        
        G = self.G
        D = self.D
        
        GtG = G.T @ G
        DtD = (D.T @ D).toarray()
        A = GtG + rho * DtD
        Gtu = G.T @ u_measured
        
        q = np.zeros(n)
        z = np.zeros(n_edges)
        w = np.zeros(n_edges)
        
        history = {'objective': [], 'tv_norm': []}
        
        for k in range(max_iter):
            # q-update
            rhs = Gtu + rho * D.T @ (z - w)
            q = np.linalg.solve(A, rhs)
            
            # z-update
            Dq = D @ q
            z = self.soft_threshold(Dq + w, alpha / rho)
            
            # w-update
            w = w + Dq - z
            
            # Convergence check
            primal_res = np.linalg.norm(Dq - z)
            objective = np.linalg.norm(G @ q - u_measured)**2 + alpha * np.sum(np.abs(Dq))
            tv_norm = np.sum(np.abs(Dq))
            
            history['objective'].append(objective)
            history['tv_norm'].append(tv_norm)
            
            if k % 20 == 0:
                print(f"    Iter {k}: obj={objective:.4e}, TV={tv_norm:.4e}")
            
            if primal_res < tol:
                print(f"  Converged at iteration {k}")
                break
        
        return q, history
    
    def _solve_tv_primal_dual(self, u_measured, alpha, tau, sigma, max_iter, tol):
        """
        TV regularization via Chambolle-Pock primal-dual algorithm.
        """
        print(f"  TV-Primal-Dual (alpha={alpha}, tau={tau}, sigma={sigma})")
        
        n = self.n_interior
        n_edges = self.D.shape[0]
        
        G = self.G
        D = self.D
        
        q = np.zeros(n)
        q_bar = np.zeros(n)
        p = np.zeros(n_edges)
        
        GtG = G.T @ G
        Gtu = G.T @ u_measured
        
        history = {'objective': [], 'tv_norm': []}
        
        for k in range(max_iter):
            # Dual update
            p = p + sigma * (D @ q_bar)
            p = np.clip(p, -alpha, alpha)
            
            # Primal update
            q_old = q.copy()
            rhs = q - tau * (D.T @ p) + tau * Gtu
            q = np.linalg.solve(np.eye(n) + tau * GtG, rhs)
            
            # Extrapolation
            q_bar = 2 * q - q_old
            
            # Convergence
            change = np.linalg.norm(q - q_old) / (np.linalg.norm(q_old) + 1e-10)
            
            Dq = D @ q
            objective = np.linalg.norm(G @ q - u_measured)**2 + alpha * np.sum(np.abs(Dq))
            tv_norm = np.sum(np.abs(Dq))
            
            history['objective'].append(objective)
            history['tv_norm'].append(tv_norm)
            
            if k % 50 == 0:
                print(f"    Iter {k}: obj={objective:.4e}, change={change:.4e}")
            
            if change < tol:
                print(f"  Converged at iteration {k}")
                break
        
        return q, history
    
    def plot_results(self, q, sources_true=None, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        ax = axes[0]
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)
        vmax = max(np.max(np.abs(q)), 0.01)
        scatter = ax.scatter(
            self.interior_coords[:, 0], self.interior_coords[:, 1],
            c=q, cmap='RdBu_r', s=40, vmin=-vmax, vmax=vmax
        )
        plt.colorbar(scatter, ax=ax, label='q')
        if sources_true:
            for (x, y), intensity in sources_true:
                marker = 'o' if intensity > 0 else 's'
                ax.plot(x, y, marker, color='black', markersize=15,
                       markerfacecolor='none', markeredgewidth=3)
        reg = self.config.get('linear', 'regularization')
        ax.set_title(f'Source Distribution ({reg})')
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
    
    def compare_regularizations(self, u_measured, sources_true=None, save_path=None):
        """Compare L1, L2, TV side by side."""
        if self.G is None:
            self.build_greens_matrix()
        if self.D is None:
            self.build_gradient_operator()
        
        print("\nComparing regularization methods...")
        
        # L2
        q_l2 = self._solve_l2(u_measured, alpha=1e-4)
        q_l2 = q_l2 - np.mean(q_l2)
        
        # L1
        q_l1 = self._solve_l1(u_measured, alpha=1e-4, max_iter=50)
        q_l1 = q_l1 - np.mean(q_l1)
        
        # TV
        q_tv, _ = self._solve_tv_admm(u_measured, alpha=1e-3, rho=1.0, max_iter=200, tol=1e-6)
        q_tv = q_tv - np.mean(q_tv)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        for col, (name, q) in enumerate([('L2', q_l2), ('L1', q_l1), ('TV', q_tv)]):
            ax = axes[0, col]
            ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
            vmax = max(np.max(np.abs(q)), 0.01)
            scatter = ax.scatter(
                self.interior_coords[:, 0], self.interior_coords[:, 1],
                c=q, cmap='RdBu_r', s=30, vmin=-vmax, vmax=vmax
            )
            plt.colorbar(scatter, ax=ax, label='q')
            if sources_true:
                for (x, y), intensity in sources_true:
                    marker = 'o' if intensity > 0 else 's'
                    ax.plot(x, y, marker, color='black', markersize=12,
                           markerfacecolor='none', markeredgewidth=2)
            ax.set_title(f'{name}')
            ax.set_aspect('equal')
            
            ax = axes[1, col]
            ax.plot(self.boundary_angles, u_measured, 'b-', linewidth=2, label='Measured')
            ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Fit')
            ax.legend()
            ax.grid(True, alpha=0.3)
            residual = np.linalg.norm(self.G @ q - u_measured)
            ax.set_title(f'Residual={residual:.4f}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return {'l2': q_l2, 'l1': q_l1, 'tv': q_tv}


# ============================================================================
# HELPER
# ============================================================================

def generate_synthetic_data(sources, noise_level=0.0, mesh_resolution=0.05):
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
    parser = argparse.ArgumentParser(description='Inverse Source Localization')
    parser.add_argument('--config', type=str, default='src/config.json')
    parser.add_argument('--no-live', action='store_true')
    parser.add_argument('--optimizer', type=str, choices=ALL_OPTIMIZERS)
    parser.add_argument('--regularization', type=str, choices=LINEAR_REGULARIZATIONS)
    parser.add_argument('--method', type=str, choices=['nonlinear', 'linear', 'both'])
    parser.add_argument('--list-optimizers', action='store_true')
    parser.add_argument('--compare-regularizations', action='store_true')
    args = parser.parse_args()
    
    if args.list_optimizers:
        list_optimizers()
        return
    
    config = Config(args.config)
    
    if args.no_live:
        config.set(False, 'visualization', 'live_plot')
    if args.optimizer:
        config.set(args.optimizer, 'nonlinear', 'optimizer')
    if args.regularization:
        config.set(args.regularization, 'linear', 'regularization')
    if args.method:
        if args.method == 'both':
            config.set(True, 'solver', 'run_both')
        else:
            config.set(args.method, 'solver', 'method')
            config.set(False, 'solver', 'run_both')
    
    config.print_summary()
    
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
    
    run_both = config.get('solver', 'run_both')
    method = config.get('solver', 'method')
    
    # Compare all regularizations
    if args.compare_regularizations:
        print("\n" + "="*60)
        print("COMPARING REGULARIZATIONS")
        print("="*60)
        
        solver = LinearInverseSolver(config)
        solver.build_greens_matrix()
        
        interp = interp1d(
            np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi]),
            np.tile(values, 3), kind='linear'
        )
        u_measured = interp(solver.boundary_angles)
        solver.u_measured = u_measured
        
        solver.compare_regularizations(u_measured, sources_true,
                                       save_path=f"{output_dir}/regularization_comparison.png")
        return
    
    if run_both or method == 'nonlinear':
        print("\n" + "="*60)
        print("NONLINEAR SOLVER")
        print("="*60)
        
        solver = NonlinearInverseSolver(config)
        solver.set_measured_data(angles, values)
        sources_rec, _ = solver.solve(sources_true=sources_true)
        solver.compare_results(sources_true, sources_rec, 
                              save_path=f"{output_dir}/inverse_nonlinear.png")
    
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
