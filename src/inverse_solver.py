"""
Inverse Solver for Source Localization
=======================================
Two approaches:
1. Linear (discrete): Sources at mesh nodes, solve linear system
2. Nonlinear (continuous): Sources anywhere, use optimization

Author: [Your Name]
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from forward_solver import (
    create_disk_mesh,
    solve_poisson_zero_neumann,
    get_boundary_values,
    create_source_function
)


# ============================================================================
# LINEAR INVERSE SOLVER (Discrete: sources at mesh nodes)
# ============================================================================

class LinearInverseSolver:
    """
    Linear inverse solver assuming sources are located at mesh nodes.
    
    This discretizes the problem:
    - Pre-compute Green's matrix G (boundary response to unit source at each node)
    - Solve: min ||G @ q - u_measured||^2 + regularization
    """
    
    def __init__(self, mesh_resolution=0.05, radius=1.0):
        """
        Initialize solver and build Green's matrix.
        """
        self.radius = radius
        self.mesh_resolution = mesh_resolution
        
        # Create mesh
        print("Creating mesh...")
        self.mesh, _, _ = create_disk_mesh(radius, mesh_resolution)
        
        # Get interior nodes (potential source locations)
        self.all_coords = self.mesh.geometry.x[:, :2]  # All node coordinates
        radii = np.sqrt(self.all_coords[:, 0]**2 + self.all_coords[:, 1]**2)
        
        # Interior nodes (exclude boundary)
        self.interior_mask = radii < 0.95 * radius
        self.interior_coords = self.all_coords[self.interior_mask]
        self.n_interior = len(self.interior_coords)
        
        print(f"Total nodes: {len(self.all_coords)}")
        print(f"Interior nodes (potential source locations): {self.n_interior}")
        
        # Boundary nodes for measurements
        self.boundary_mask = radii > 0.9 * radius
        self.boundary_coords = self.all_coords[self.boundary_mask]
        self.n_boundary = len(self.boundary_coords)
        self.boundary_angles = np.arctan2(self.boundary_coords[:, 1], self.boundary_coords[:, 0])
        
        # Sort boundary by angle
        sort_idx = np.argsort(self.boundary_angles)
        self.boundary_angles = self.boundary_angles[sort_idx]
        self.boundary_indices = np.where(self.boundary_mask)[0][sort_idx]
        
        print(f"Boundary nodes (measurement points): {self.n_boundary}")
        
        # Green's matrix (built lazily)
        self.G = None
        
    def build_greens_matrix(self, sigma=0.03):
        """
        Build Green's matrix by solving forward problem for unit source at each interior node.
        
        G[i, j] = boundary value at point i due to unit source at interior node j
        
        This is expensive but only done once!
        """
        print(f"\nBuilding Green's matrix ({self.n_boundary} x {self.n_interior})...")
        print("This requires solving", self.n_interior, "forward problems...")
        
        self.G = np.zeros((self.n_boundary, self.n_interior))
        
        for j in range(self.n_interior):
            if j % 50 == 0:
                print(f"  Computing column {j}/{self.n_interior}...")
            
            # Unit source at interior node j
            x_j, y_j = self.interior_coords[j]
            sources = [((x_j, y_j), 1.0), ((0, 0), -1.0)]  # Balance with sink at origin
            
            # Solve forward problem
            u = solve_poisson_zero_neumann(self.mesh, sources)
            
            # Extract boundary values
            _, boundary_values = get_boundary_values(u)
            
            # Store in Green's matrix (need to align with our boundary ordering)
            angles_computed, values_computed = get_boundary_values(u)
            interp = interp1d(
                np.concatenate([angles_computed - 2*np.pi, angles_computed, angles_computed + 2*np.pi]),
                np.concatenate([values_computed, values_computed, values_computed]),
                kind='linear'
            )
            self.G[:, j] = interp(self.boundary_angles)
        
        print("Green's matrix built!")
        return self.G
    
    def solve(self, u_measured, regularization='l2', alpha=1e-6):
        """
        Solve the linear inverse problem.
        
        Parameters
        ----------
        u_measured : np.ndarray
            Measured boundary values (at self.boundary_angles)
        regularization : str
            'l2' (Tikhonov) or 'l1' (sparsity-promoting)
        alpha : float
            Regularization parameter
        
        Returns
        -------
        q : np.ndarray
            Source intensities at interior nodes
        sources : list
            Non-zero sources as [((x, y), intensity), ...]
        """
        if self.G is None:
            self.build_greens_matrix()
        
        print(f"\nSolving linear inverse problem (regularization={regularization}, alpha={alpha})...")
        
        if regularization == 'l2':
            # Tikhonov regularization: min ||Gq - u||^2 + alpha * ||q||^2
            # Normal equations: (G'G + alpha*I) q = G' u
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q = np.linalg.solve(GtG + alpha * np.eye(self.n_interior), Gtu)
            
        elif regularization == 'l1':
            # L1 regularization (sparsity): use iterative solver
            # Approximate with iteratively reweighted least squares
            q = self._solve_l1(u_measured, alpha)
        
        else:
            raise ValueError(f"Unknown regularization: {regularization}")
        
        # Enforce zero sum (Neumann compatibility)
        q = q - np.mean(q)
        
        # Extract significant sources
        threshold = 0.1 * np.max(np.abs(q))
        sources = []
        for j in range(self.n_interior):
            if np.abs(q[j]) > threshold:
                x, y = self.interior_coords[j]
                sources.append(((x, y), q[j]))
        
        print(f"Found {len(sources)} significant sources (threshold={threshold:.4f})")
        
        return q, sources
    
    def _solve_l1(self, u_measured, alpha, max_iter=50):
        """L1 regularization via iteratively reweighted least squares."""
        q = np.zeros(self.n_interior)
        epsilon = 1e-4
        
        for iteration in range(max_iter):
            # Weights for reweighted least squares
            weights = 1.0 / (np.abs(q) + epsilon)
            W = np.diag(weights)
            
            # Solve weighted problem
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            
            # Check convergence
            if np.linalg.norm(q_new - q) < 1e-6:
                break
            q = q_new
        
        return q
    
    def plot_results(self, q, sources_true=None, save_path=None):
        """Plot the recovered source distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Source intensity map
        ax = axes[0]
        self._plot_mesh(ax, show_mesh=True)
        
        # Color interior nodes by intensity
        scatter = ax.scatter(
            self.interior_coords[:, 0],
            self.interior_coords[:, 1],
            c=q,
            cmap='RdBu_r',
            s=30,
            vmin=-np.max(np.abs(q)),
            vmax=np.max(np.abs(q))
        )
        plt.colorbar(scatter, ax=ax, label='Source intensity q')
        
        # Mark true sources if provided
        if sources_true:
            for (x, y), intensity in sources_true:
                marker = 'o' if intensity > 0 else 's'
                ax.plot(x, y, marker, color='black', markersize=15, 
                       markerfacecolor='none', markeredgewidth=3, label='True')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Recovered Source Distribution (Linear Method)')
        ax.set_aspect('equal')
        
        # Plot 2: Boundary fit
        ax = axes[1]
        if hasattr(self, 'u_measured'):
            ax.plot(self.boundary_angles, self.u_measured, 'b-', linewidth=2, label='Measured')
        u_reconstructed = self.G @ q
        ax.plot(self.boundary_angles, u_reconstructed, 'r--', linewidth=2, label='Reconstructed')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Boundary value')
        ax.set_title('Boundary Data Fit')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_mesh(self, ax, show_mesh=True):
        """Helper to plot the mesh."""
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        
        if show_mesh:
            tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
            ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)


# ============================================================================
# NONLINEAR INVERSE SOLVER (Continuous: sources anywhere)
# ============================================================================

class NonlinearInverseSolver:
    """
    Nonlinear inverse solver - sources can be anywhere in the domain.
    Uses optimization to find (x, y, q) for each source.
    """
    
    def __init__(self, n_sources, radius=1.0, mesh_resolution=0.05):
        """
        Parameters
        ----------
        n_sources : int
            Number of sources to recover (must know this a priori)
        """
        self.n_sources = n_sources
        self.radius = radius
        self.mesh_resolution = mesh_resolution
        
        print("Creating mesh for inverse solver...")
        self.mesh, _, _ = create_disk_mesh(radius, mesh_resolution)
        
        self.angles_measured = None
        self.values_measured = None
        self.interpolator = None
        self.history = []
        
    def set_measured_data(self, angles, values):
        """Set the measured boundary data."""
        sort_idx = np.argsort(angles)
        self.angles_measured = angles[sort_idx]
        self.values_measured = values[sort_idx]
        
        # Periodic interpolator
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
            x = params[3*i]
            y = params[3*i + 1]
            q = params[3*i + 2]
            sources.append(((x, y), q))
        
        # Last intensity computed for balance
        x_last = params[3*(n-1)]
        y_last = params[3*(n-1) + 1]
        q_last = -sum(q for _, q in sources)
        sources.append(((x_last, y_last), q_last))
        
        return sources
    
    def objective(self, params):
        """Objective: ||u_measured - u_computed||²"""
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
            
            self.history.append({'misfit': misfit, 'sources': sources})
            if len(self.history) % 20 == 0:
                print(f"  Iteration {len(self.history)}: misfit = {misfit:.6e}")
            
            return misfit
        except:
            return 1e10
    
    def solve(self, method='differential_evolution', **kwargs):
        """Solve using optimization."""
        self.history = []
        n = self.n_sources
        r_max = 0.9 * self.radius
        
        bounds = []
        for i in range(n):
            bounds.append((-r_max, r_max))  # x
            bounds.append((-r_max, r_max))  # y
            if i < n - 1:
                bounds.append((-5.0, 5.0))  # q
        
        print(f"\nRunning {method} ({n} sources)...")
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective, bounds,
                maxiter=kwargs.get('maxiter', 100),
                popsize=kwargs.get('popsize', 15),
                seed=kwargs.get('seed', 42),
                workers=1, updating='deferred', disp=True
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        sources_recovered = self.params_to_sources(result.x)
        print(f"\nFinal misfit: {result.fun:.6e}")
        
        return sources_recovered, result
    
    def compare_results(self, sources_true, sources_recovered, save_path=None):
        """Compare true and recovered sources with mesh visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Get mesh triangulation
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        
        # Plot 1: True sources
        ax = axes[0]
        u_true = solve_poisson_zero_neumann(self.mesh, sources_true)
        u_vals = u_true.x.array[:len(coords)] if len(u_true.x.array) != len(coords) else u_true.x.array
        
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)  # Mesh
        tcf = ax.tricontourf(tri, u_vals, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(tcf, ax=ax, label='u')
        
        for (x, y), q in sources_true:
            color, marker = ('red', '+') if q > 0 else ('blue', '*')
            ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
        ax.set_title('True Sources')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot 2: Recovered sources
        ax = axes[1]
        u_rec = solve_poisson_zero_neumann(self.mesh, sources_recovered)
        u_vals = u_rec.x.array[:len(coords)] if len(u_rec.x.array) != len(coords) else u_rec.x.array
        
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)  # Mesh
        tcf = ax.tricontourf(tri, u_vals, levels=50, cmap='viridis', alpha=0.8)
        plt.colorbar(tcf, ax=ax, label='u')
        
        for (x, y), q in sources_recovered:
            color, marker = ('red', '+') if q > 0 else ('blue', '*')
            ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)
        ax.set_title('Recovered Sources')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Plot 3: Boundary comparison
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
        print("TRUE vs RECOVERED SOURCES")
        print("="*60)
        print("\nTrue:")
        for i, ((x, y), q) in enumerate(sources_true):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")
        print("\nRecovered:")
        for i, ((x, y), q) in enumerate(sources_recovered):
            print(f"  {i+1}: ({x:+.4f}, {y:+.4f}), q={q:+.4f}")


# ============================================================================
# HELPER FUNCTIONS
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

if __name__ == "__main__":
    print("="*60)
    print("Inverse Problem: Source Localization")
    print("="*60)
    
    # True sources
    sources_true = [
        ((-0.3, 0.4), 1.0),
        ((0.5, 0.3), 1.0),
        ((-0.4, -0.4), -1.0),
        ((0.3, -0.5), -1.0),
    ]
    
    print("\nTrue sources:")
    for i, ((x, y), q) in enumerate(sources_true):
        print(f"  {i+1}: ({x:+.2f}, {y:+.2f}), q={q:+.1f}")
    
    # Generate data
    print("\nGenerating synthetic data...")
    angles, values = generate_synthetic_data(sources_true, noise_level=0.001)
    
    # =========================================
    # Method 1: Nonlinear (continuous search)
    # =========================================
    print("\n" + "="*60)
    print("METHOD 1: Nonlinear (continuous source locations)")
    print("="*60)
    
    solver_nl = NonlinearInverseSolver(n_sources=4, mesh_resolution=0.05)
    solver_nl.set_measured_data(angles, values)
    sources_nl, _ = solver_nl.solve(method='differential_evolution', maxiter=50, popsize=10)
    solver_nl.compare_results(sources_true, sources_nl, save_path="results/inverse_nonlinear.png")
    
    # =========================================
    # Method 2: Linear (discrete mesh nodes)
    # =========================================
    print("\n" + "="*60)
    print("METHOD 2: Linear (sources at mesh nodes)")
    print("="*60)
    
    solver_lin = LinearInverseSolver(mesh_resolution=0.08)  # Coarser mesh for speed
    solver_lin.build_greens_matrix()
    
    # Interpolate measured data to solver's boundary angles
    interp = interp1d(
        np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi]),
        np.tile(values, 3), kind='linear'
    )
    u_measured = interp(solver_lin.boundary_angles)
    solver_lin.u_measured = u_measured
    
    q, sources_lin = solver_lin.solve(u_measured, regularization='l1', alpha=1e-4)
    solver_lin.plot_results(q, sources_true, save_path="results/inverse_linear.png")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
