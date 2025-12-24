"""
Inverse Solver for Source Localization
=======================================
Given boundary measurements, recover source locations and intensities.

Approach: Optimization-based
- Minimize ||u_measured - u_computed||² 
- Subject to: sum of intensities = 0 (Neumann compatibility)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from forward_solver import (
    create_disk_mesh,
    solve_poisson_zero_neumann,
    get_boundary_values,
    plot_solution
)


class InverseSolver:
    """
    Inverse solver for point source localization from boundary data.
    """
    
    def __init__(self, n_sources, radius=1.0, mesh_resolution=0.05):
        """
        Initialize the inverse solver.
        
        Parameters
        ----------
        n_sources : int
            Number of sources to recover (must know this a priori)
        radius : float
            Domain radius
        mesh_resolution : float
            Mesh resolution for forward solves
        """
        self.n_sources = n_sources
        self.radius = radius
        self.mesh_resolution = mesh_resolution
        
        # Create mesh once (reused for all forward solves)
        print("Creating mesh for inverse solver...")
        self.mesh, _, _ = create_disk_mesh(radius, mesh_resolution)
        
        # Storage for measured data
        self.angles_measured = None
        self.values_measured = None
        self.interpolator = None
        
        # Optimization history
        self.history = []
        
    def set_measured_data(self, angles, values):
        """
        Set the measured boundary data.
        
        Parameters
        ----------
        angles : np.ndarray
            Angles (radians) of measurement points
        values : np.ndarray
            Measured solution values at those angles
        """
        # Sort by angle
        sort_idx = np.argsort(angles)
        self.angles_measured = angles[sort_idx]
        self.values_measured = values[sort_idx]
        
        # Create interpolator for comparing at arbitrary angles
        # Extend periodically for interpolation
        angles_ext = np.concatenate([
            self.angles_measured - 2*np.pi,
            self.angles_measured,
            self.angles_measured + 2*np.pi
        ])
        values_ext = np.concatenate([
            self.values_measured,
            self.values_measured,
            self.values_measured
        ])
        self.interpolator = interp1d(angles_ext, values_ext, kind='linear')
        
    def params_to_sources(self, params):
        """
        Convert flat parameter array to sources list.
        
        Parameters format: [x1, y1, q1, x2, y2, q2, ...]
        Last intensity is computed to ensure sum = 0
        """
        sources = []
        n = self.n_sources
        
        for i in range(n - 1):
            x = params[3*i]
            y = params[3*i + 1]
            q = params[3*i + 2]
            sources.append(((x, y), q))
        
        # Last source: position from params, intensity computed for balance
        x_last = params[3*(n-1)]
        y_last = params[3*(n-1) + 1]
        q_last = -sum(q for _, q in sources)  # Enforce sum = 0
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def sources_to_params(self, sources):
        """Convert sources list to flat parameter array."""
        params = []
        for i, ((x, y), q) in enumerate(sources):
            params.extend([x, y])
            if i < len(sources) - 1:  # Don't include last intensity
                params.append(q)
        return np.array(params)
    
    def objective(self, params):
        """
        Objective function: ||u_measured - u_computed||²
        """
        # Check if sources are inside domain
        sources = self.params_to_sources(params)
        for (x, y), q in sources:
            if x**2 + y**2 >= (0.95 * self.radius)**2:
                return 1e10  # Penalty for sources outside domain
        
        try:
            # Solve forward problem
            u = solve_poisson_zero_neumann(self.mesh, sources)
            
            # Get boundary values
            angles_comp, values_comp = get_boundary_values(u)
            
            # Interpolate measured data at computed angles
            values_measured_interp = self.interpolator(angles_comp)
            
            # Compute misfit
            misfit = np.sum((values_comp - values_measured_interp)**2)
            
            # Store history
            self.history.append({
                'params': params.copy(),
                'misfit': misfit,
                'sources': sources
            })
            
            if len(self.history) % 10 == 0:
                print(f"  Iteration {len(self.history)}: misfit = {misfit:.6e}")
            
            return misfit
            
        except Exception as e:
            print(f"Forward solve failed: {e}")
            return 1e10
    
    def solve(self, initial_guess=None, method='differential_evolution', **kwargs):
        """
        Solve the inverse problem.
        
        Parameters
        ----------
        initial_guess : list of sources, optional
            Initial guess for source configuration
        method : str
            'differential_evolution' (global) or 'L-BFGS-B' (local)
        **kwargs : dict
            Additional arguments for the optimizer
        
        Returns
        -------
        sources : list
            Recovered sources as [((x, y), intensity), ...]
        result : OptimizeResult
            Full optimization result
        """
        self.history = []
        
        n = self.n_sources
        
        # Parameter bounds: x, y in disk, intensities unbounded but reasonable
        r_max = 0.9 * self.radius
        bounds = []
        for i in range(n):
            bounds.append((-r_max, r_max))  # x
            bounds.append((-r_max, r_max))  # y
            if i < n - 1:  # Last intensity is computed
                bounds.append((-5.0, 5.0))  # q
        
        if method == 'differential_evolution':
            print(f"\nRunning differential evolution ({n} sources)...")
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=kwargs.get('maxiter', 100),
                popsize=kwargs.get('popsize', 15),
                tol=kwargs.get('tol', 1e-6),
                seed=kwargs.get('seed', 42),
                workers=1,  # Serial for stability
                updating='deferred',
                disp=True
            )
        
        elif method == 'L-BFGS-B':
            if initial_guess is None:
                # Random initial guess
                x0 = []
                for i in range(n):
                    angle = 2 * np.pi * i / n
                    r = 0.5 * self.radius
                    x0.extend([r * np.cos(angle), r * np.sin(angle)])
                    if i < n - 1:
                        x0.append(1.0 if i % 2 == 0 else -1.0)
                x0 = np.array(x0)
            else:
                x0 = self.sources_to_params(initial_guess)
            
            print(f"\nRunning L-BFGS-B ({n} sources)...")
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': kwargs.get('maxiter', 200), 'disp': True}
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Extract final sources
        sources_recovered = self.params_to_sources(result.x)
        
        print(f"\nOptimization complete!")
        print(f"  Final misfit: {result.fun:.6e}")
        print(f"  Iterations: {len(self.history)}")
        
        return sources_recovered, result
    
    def compare_results(self, sources_true, sources_recovered, save_path=None):
        """
        Compare true and recovered sources visually.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: True solution
        ax = axes[0]
        u_true = solve_poisson_zero_neumann(self.mesh, sources_true)
        self._plot_solution_on_ax(ax, u_true, sources_true, "True Sources")
        
        # Plot 2: Recovered solution
        ax = axes[1]
        u_recovered = solve_poisson_zero_neumann(self.mesh, sources_recovered)
        self._plot_solution_on_ax(ax, u_recovered, sources_recovered, "Recovered Sources")
        
        # Plot 3: Boundary comparison
        ax = axes[2]
        angles_true, values_true = get_boundary_values(u_true)
        angles_rec, values_rec = get_boundary_values(u_recovered)
        
        ax.plot(angles_true, values_true, 'b-', linewidth=2, label='True')
        ax.plot(angles_rec, values_rec, 'r--', linewidth=2, label='Recovered')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Solution Value')
        ax.set_title('Boundary Values Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print source comparison
        print("\n" + "="*60)
        print("Source Comparison")
        print("="*60)
        print("\nTrue sources:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q = {q:+.4f}")
        
        print("\nRecovered sources:")
        for i, ((x, y), q) in enumerate(sources_recovered):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q = {q:+.4f}")
        
        # Compute errors (match sources by proximity)
        print("\nReconstruction errors:")
        total_pos_error = 0
        total_int_error = 0
        for i, ((x_r, y_r), q_r) in enumerate(sources_recovered):
            # Find closest true source
            min_dist = float('inf')
            closest = None
            for j, ((x_t, y_t), q_t) in enumerate(sources_true):
                dist = np.sqrt((x_r - x_t)**2 + (y_r - y_t)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest = (x_t, y_t, q_t)
            
            pos_err = min_dist
            int_err = abs(q_r - closest[2])
            total_pos_error += pos_err
            total_int_error += int_err
            print(f"  Source {i+1}: position error = {pos_err:.4f}, intensity error = {int_err:.4f}")
        
        print(f"\nTotal position error: {total_pos_error:.4f}")
        print(f"Total intensity error: {total_int_error:.4f}")
    
    def _plot_solution_on_ax(self, ax, u, sources, title):
        """Helper to plot solution on a given axis."""
        from matplotlib.tri import Triangulation
        
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        
        u_values = u.x.array if len(u.x.array) == len(coords) else u.x.array[:len(coords)]
        
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        tcf = ax.tricontourf(tri, u_values, levels=50, cmap='viridis')
        plt.colorbar(tcf, ax=ax, label='u')
        
        for (x, y), q in sources:
            color = 'red' if q > 0 else 'blue'
            marker = '+' if q > 0 else '*'
            ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')


def generate_synthetic_data(sources, noise_level=0.0, mesh_resolution=0.05):
    """
    Generate synthetic boundary measurements for testing.
    
    Parameters
    ----------
    sources : list
        True source configuration
    noise_level : float
        Standard deviation of Gaussian noise to add
    mesh_resolution : float
        Mesh resolution
    
    Returns
    -------
    angles, values : np.ndarray
        Boundary measurement data
    """
    mesh, _, _ = create_disk_mesh(resolution=mesh_resolution)
    u = solve_poisson_zero_neumann(mesh, sources)
    angles, values = get_boundary_values(u)
    
    if noise_level > 0:
        values = values + np.random.normal(0, noise_level, len(values))
    
    return angles, values


# ============================================================================
# MAIN: Example usage
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Inverse Problem: Source Localization from Boundary Data")
    print("="*60)
    
    # Define true sources (unknown to inverse solver)
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    print("\nTrue sources (to be recovered):")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.2f}, {y:+.2f}), intensity = {q:+.1f}")
    
    # Generate synthetic boundary measurements
    print("\nGenerating synthetic boundary data...")
    angles, values = generate_synthetic_data(sources_true, noise_level=0.001)
    print(f"Generated {len(angles)} boundary measurements")
    
    # Create inverse solver
    n_sources = len(sources_true)  # Assume we know the number of sources
    solver = InverseSolver(n_sources=n_sources, mesh_resolution=0.05)
    solver.set_measured_data(angles, values)
    
    # Solve inverse problem
    sources_recovered, result = solver.solve(
        method='differential_evolution',
        maxiter=50,
        popsize=10,
        seed=42
    )
    
    # Compare results
    solver.compare_results(sources_true, sources_recovered, save_path="results/inverse_comparison.png")
    
    print("\nDone!")
