"""
Inverse Solver for Source Localization
=======================================
Configurable via JSON file.

Usage:
    python inverse_solver.py                    # Use default config.json
    python inverse_solver.py --config my.json   # Use custom config
    python inverse_solver.py --no-live          # Disable live plotting
"""

import numpy as np
import json
import argparse
from pathlib import Path
from scipy.optimize import minimize, differential_evolution, basinhopping, dual_annealing
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from forward_solver import (
    create_disk_mesh,
    solve_poisson_zero_neumann,
    get_boundary_values,
)


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class Config:
    """Load and manage configuration from JSON file."""
    
    DEFAULT_CONFIG = {
        "problem": {
            "n_sources": 4,
            "radius": 1.0,
            "noise_level": 0.001
        },
        "mesh": {
            "resolution_forward": 0.05,
            "resolution_inverse_linear": 0.08
        },
        "sources_true": [
            {"x": -0.3, "y": 0.4, "intensity": 1.0},
            {"x": 0.5, "y": 0.3, "intensity": 1.0},
            {"x": -0.4, "y": -0.4, "intensity": -1.0},
            {"x": 0.3, "y": -0.5, "intensity": -1.0}
        ],
        "solver": {
            "method": "nonlinear",
            "run_both": True
        },
        "nonlinear": {
            "optimizer": "differential_evolution",
            "differential_evolution": {
                "maxiter": 50,
                "popsize": 10,
                "tol": 1e-6,
                "seed": 42
            }
        },
        "linear": {
            "regularization": "l1",
            "alpha": 1e-4,
            "l1_max_iter": 50
        },
        "visualization": {
            "live_plot": True,
            "update_interval": 5,
            "save_results": True,
            "output_dir": "results"
        }
    }
    
    def __init__(self, config_path=None):
        """Load config from file or use defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            print(f"Loading config from: {config_path}")
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self._deep_update(self.config, user_config)
        else:
            print("Using default configuration")
    
    def _deep_update(self, base, update):
        """Recursively update nested dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys, default=None):
        """Get nested config value: config.get('solver', 'method')"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_sources_true(self):
        """Convert sources_true from config format to list of tuples."""
        sources_config = self.get('sources_true', default=[])
        return [((s['x'], s['y']), s['intensity']) for s in sources_config]
    
    def save(self, path):
        """Save current config to file."""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Config saved to: {path}")
    
    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Problem: {self.get('problem', 'n_sources')} sources, "
              f"radius={self.get('problem', 'radius')}, "
              f"noise={self.get('problem', 'noise_level')}")
        print(f"Solver method: {self.get('solver', 'method')}")
        print(f"Run both: {self.get('solver', 'run_both')}")
        print(f"Nonlinear optimizer: {self.get('nonlinear', 'optimizer')}")
        print(f"Linear regularization: {self.get('linear', 'regularization')}, "
              f"alpha={self.get('linear', 'alpha')}")
        print(f"Live plotting: {self.get('visualization', 'live_plot')}")
        print("="*60 + "\n")


# ============================================================================
# LIVE VISUALIZATION CLASS
# ============================================================================

class LivePlotter:
    """Live visualization during optimization."""
    
    def __init__(self, mesh, sources_true=None, update_interval=5):
        self.mesh = mesh
        self.sources_true = sources_true
        self.update_interval = update_interval
        self.iteration = 0
        self.history = []
        
        # Setup mesh triangulation
        coords = mesh.geometry.x
        mesh.topology.create_connectivity(2, 0)
        cells = mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        self.tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        self.coords = coords
        
        # Initialize figure
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Inverse Problem - Live Optimization')
        self._init_plots()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        
    def _init_plots(self):
        """Initialize empty plots."""
        self.ax_solution = self.axes[0]
        self.ax_solution.set_title('Current Solution')
        self.ax_solution.set_aspect('equal')
        self.ax_solution.set_xlim(-1.1, 1.1)
        self.ax_solution.set_ylim(-1.1, 1.1)
        
        if self.sources_true:
            for (x, y), q in self.sources_true:
                color = 'green' if q > 0 else 'purple'
                self.ax_solution.plot(x, y, 'o', color=color, markersize=20,
                                     markerfacecolor='none', markeredgewidth=2)
        
        self.ax_boundary = self.axes[1]
        self.ax_boundary.set_title('Boundary Fit')
        self.ax_boundary.grid(True, alpha=0.3)
        
        self.ax_convergence = self.axes[2]
        self.ax_convergence.set_title('Convergence')
        self.ax_convergence.set_yscale('log')
        self.ax_convergence.grid(True, alpha=0.3)
        
    def update(self, sources, misfit, angles_measured=None, values_measured=None):
        """Update the visualization."""
        self.iteration += 1
        self.history.append(misfit)
        
        if self.iteration % self.update_interval != 0:
            return
        
        # Clear and redraw
        for ax in self.axes:
            ax.clear()
        
        self.ax_solution = self.axes[0]
        self.ax_solution.set_title(f'Current Solution (iter {self.iteration})')
        self.ax_solution.set_aspect('equal')
        
        self.ax_boundary = self.axes[1]
        self.ax_boundary.set_title('Boundary Fit')
        self.ax_boundary.grid(True, alpha=0.3)
        
        self.ax_convergence = self.axes[2]
        self.ax_convergence.set_title(f'Convergence (misfit={misfit:.2e})')
        self.ax_convergence.set_yscale('log')
        self.ax_convergence.grid(True, alpha=0.3)
        
        try:
            u = solve_poisson_zero_neumann(self.mesh, sources)
            u_vals = u.x.array[:len(self.coords)] if len(u.x.array) != len(self.coords) else u.x.array
            
            # Solution field
            self.ax_solution.triplot(self.tri, 'k-', linewidth=0.2, alpha=0.2)
            self.ax_solution.tricontourf(self.tri, u_vals, levels=30, cmap='viridis', alpha=0.8)
            
            # True sources
            if self.sources_true:
                for (x, y), q in self.sources_true:
                    color = 'green' if q > 0 else 'purple'
                    self.ax_solution.plot(x, y, 'o', color=color, markersize=18,
                                         markerfacecolor='none', markeredgewidth=3)
            
            # Current sources
            for (x, y), q in sources:
                color = 'red' if q > 0 else 'blue'
                marker = '+' if q > 0 else '*'
                self.ax_solution.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
            
            # Boundary comparison
            angles_comp, values_comp = get_boundary_values(u)
            if angles_measured is not None:
                self.ax_boundary.plot(angles_measured, values_measured, 'b-', 
                                     linewidth=2, label='Measured', alpha=0.7)
            self.ax_boundary.plot(angles_comp, values_comp, 'r--', linewidth=2, label='Current')
            self.ax_boundary.legend(loc='upper right')
            
        except Exception as e:
            self.ax_solution.text(0, 0, f'Error: {e}', ha='center')
        
        # Convergence
        if len(self.history) > 1:
            self.ax_convergence.plot(range(1, len(self.history) + 1), self.history, 'b.-')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def finish(self, save_path=None):
        """Finalize and optionally save."""
        plt.ioff()
        if save_path:
            self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    def close(self):
        plt.close(self.fig)


# ============================================================================
# NONLINEAR INVERSE SOLVER
# ============================================================================

class NonlinearInverseSolver:
    """Nonlinear inverse solver - sources can be anywhere."""
    
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
    
    def objective(self, params):
        """Objective function."""
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
                self.live_plotter.update(sources, misfit, 
                                        self.angles_measured, self.values_measured)
            
            if len(self.history) % 20 == 0:
                print(f"  Iteration {len(self.history)}: misfit = {misfit:.6e}")
            
            return misfit
        except:
            return 1e10
    
    def solve(self, sources_true=None):
        """Solve using config settings."""
        self.history = []
        
        # Get settings from config
        optimizer = self.config.get('nonlinear', 'optimizer')
        opt_params = self.config.get('nonlinear', optimizer, default={})
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
        
        print(f"\nRunning {optimizer} ({n} sources)...")
        
        # Run optimizer
        if optimizer == 'differential_evolution':
            result = differential_evolution(
                self.objective, bounds,
                maxiter=opt_params.get('maxiter', 100),
                popsize=opt_params.get('popsize', 15),
                tol=opt_params.get('tol', 1e-6),
                seed=opt_params.get('seed', None),
                mutation=opt_params.get('mutation', (0.5, 1)),
                recombination=opt_params.get('recombination', 0.7),
                workers=1, updating='deferred', disp=True
            )
            
        elif optimizer == 'basinhopping':
            # Initial guess
            x0 = []
            for i in range(n):
                angle = 2 * np.pi * i / n
                r = 0.5 * self.radius
                x0.extend([r * np.cos(angle), r * np.sin(angle)])
                if i < n - 1:
                    x0.append(1.0 if i % 2 == 0 else -1.0)
            
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
            result = basinhopping(
                self.objective, x0,
                niter=opt_params.get('niter', 100),
                T=opt_params.get('T', 1.0),
                stepsize=opt_params.get('stepsize', 0.5),
                seed=opt_params.get('seed', None),
                minimizer_kwargs=minimizer_kwargs
            )
            
        elif optimizer == 'dual_annealing':
            result = dual_annealing(
                self.objective, bounds,
                maxiter=opt_params.get('maxiter', 100),
                initial_temp=opt_params.get('initial_temp', 5230.0),
                restart_temp_ratio=opt_params.get('restart_temp_ratio', 2e-5),
                seed=opt_params.get('seed', None)
            )
            
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        sources_recovered = self.params_to_sources(result.x)
        print(f"\nFinal misfit: {result.fun:.6e}")
        
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


# ============================================================================
# LINEAR INVERSE SOLVER
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
        
        print(f"Interior nodes: {self.n_interior}, Boundary nodes: {self.n_boundary}")
        
        self.G = None
        self.live_plotter = None
        
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
        return self.G
    
    def solve(self, u_measured, sources_true=None):
        """Solve using config settings."""
        if self.G is None:
            self.build_greens_matrix()
        
        reg = self.config.get('linear', 'regularization')
        alpha = self.config.get('linear', 'alpha')
        live_plot = self.config.get('visualization', 'live_plot')
        update_interval = self.config.get('visualization', 'update_interval')
        
        print(f"\nSolving (regularization={reg}, alpha={alpha})...")
        
        if live_plot:
            self.live_plotter = LivePlotter(
                self.mesh, sources_true=sources_true, update_interval=update_interval
            )
        
        if reg == 'l2':
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q = np.linalg.solve(GtG + alpha * np.eye(self.n_interior), Gtu)
            
        elif reg == 'l1':
            max_iter = self.config.get('linear', 'l1_max_iter', default=50)
            q = self._solve_l1(u_measured, alpha, max_iter, live_plot)
        else:
            raise ValueError(f"Unknown regularization: {reg}")
        
        q = q - np.mean(q)
        
        threshold = 0.1 * np.max(np.abs(q))
        sources = [((self.interior_coords[j, 0], self.interior_coords[j, 1]), q[j]) 
                   for j in range(self.n_interior) if np.abs(q[j]) > threshold]
        
        print(f"Found {len(sources)} significant sources")
        
        if live_plot and self.live_plotter:
            output_dir = self.config.get('visualization', 'output_dir')
            self.live_plotter.finish(save_path=f"{output_dir}/inverse_linear_live.png")
        
        return q, sources
    
    def _solve_l1(self, u_measured, alpha, max_iter, live_plot):
        """L1 via IRLS."""
        q = np.zeros(self.n_interior)
        epsilon = 1e-4
        
        for iteration in range(max_iter):
            weights = 1.0 / (np.abs(q) + epsilon)
            W = np.diag(weights)
            
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            
            if live_plot and self.live_plotter:
                threshold = 0.05 * np.max(np.abs(q_new))
                current_sources = [((self.interior_coords[j, 0], self.interior_coords[j, 1]), q_new[j]) 
                                  for j in range(self.n_interior) if np.abs(q_new[j]) > threshold]
                misfit = np.linalg.norm(self.G @ q_new - u_measured)**2
                self.live_plotter.update(current_sources, misfit, self.boundary_angles, u_measured)
            
            if np.linalg.norm(q_new - q) < 1e-6:
                print(f"  Converged at iteration {iteration}")
                break
            q = q_new
        
        return q
    
    def plot_results(self, q, sources_true=None, save_path=None):
        """Plot results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)
        
        scatter = ax.scatter(
            self.interior_coords[:, 0], self.interior_coords[:, 1],
            c=q, cmap='RdBu_r', s=30,
            vmin=-np.max(np.abs(q)), vmax=np.max(np.abs(q))
        )
        plt.colorbar(scatter, ax=ax, label='Source intensity q')
        
        if sources_true:
            for (x, y), intensity in sources_true:
                marker = 'o' if intensity > 0 else 's'
                ax.plot(x, y, marker, color='black', markersize=15,
                       markerfacecolor='none', markeredgewidth=3)
        
        ax.set_title('Recovered Source Distribution (Linear)')
        ax.set_aspect('equal')
        
        ax = axes[1]
        if hasattr(self, 'u_measured'):
            ax.plot(self.boundary_angles, self.u_measured, 'b-', linewidth=2, label='Measured')
        ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Reconstructed')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Boundary value')
        ax.set_title('Boundary Data Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
    # Parse arguments
    parser = argparse.ArgumentParser(description='Inverse Source Localization')
    parser.add_argument('--config', type=str, default='src/config.json',
                        help='Path to config JSON file')
    parser.add_argument('--no-live', action='store_true',
                        help='Disable live plotting')
    parser.add_argument('--method', type=str, choices=['nonlinear', 'linear', 'both'],
                        help='Override solver method')
    args = parser.parse_args()
    
    # Load config
    config = Config(args.config)
    
    # Override from command line
    if args.no_live:
        config.config['visualization']['live_plot'] = False
    if args.method:
        if args.method == 'both':
            config.config['solver']['run_both'] = True
        else:
            config.config['solver']['method'] = args.method
            config.config['solver']['run_both'] = False
    
    config.print_summary()
    
    # Get true sources and generate data
    sources_true = config.get_sources_true()
    print("True sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.2f}, {y:+.2f}), q={q:+.1f}")
    
    noise = config.get('problem', 'noise_level')
    resolution = config.get('mesh', 'resolution_forward')
    print(f"\nGenerating synthetic data (noise={noise})...")
    angles, values = generate_synthetic_data(sources_true, noise_level=noise, 
                                             mesh_resolution=resolution)
    
    output_dir = config.get('visualization', 'output_dir')
    Path(output_dir).mkdir(exist_ok=True)
    
    # Run solvers
    run_both = config.get('solver', 'run_both')
    method = config.get('solver', 'method')
    
    if run_both or method == 'nonlinear':
        print("\n" + "="*60)
        print("NONLINEAR SOLVER")
        print("="*60)
        
        solver_nl = NonlinearInverseSolver(config)
        solver_nl.set_measured_data(angles, values)
        sources_nl, _ = solver_nl.solve(sources_true=sources_true)
        solver_nl.compare_results(sources_true, sources_nl, 
                                 save_path=f"{output_dir}/inverse_nonlinear.png")
    
    if run_both or method == 'linear':
        print("\n" + "="*60)
        print("LINEAR SOLVER")
        print("="*60)
        
        solver_lin = LinearInverseSolver(config)
        solver_lin.build_greens_matrix()
        
        interp = interp1d(
            np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi]),
            np.tile(values, 3), kind='linear'
        )
        u_measured = interp(solver_lin.boundary_angles)
        solver_lin.u_measured = u_measured
        
        q, sources_lin = solver_lin.solve(u_measured, sources_true=sources_true)
        solver_lin.plot_results(q, sources_true, save_path=f"{output_dir}/inverse_linear.png")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
