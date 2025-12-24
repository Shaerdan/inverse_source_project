"""
Linear Inverse Solver with Total Variation Regularization
==========================================================

TV regularization: min_q ||Gq - u||^2 + α * TV(q)

Where TV(q) = Σ_{edges} |q_i - q_j|  (anisotropic TV on mesh)

Solved using ADMM (Alternating Direction Method of Multipliers):
    min_q,z  ||Gq - u||^2 + α||z||_1
    s.t.     Dq = z

where D is the discrete gradient operator (edge differences).

ADMM iterations:
    1. q-update: (G'G + ρD'D)q = G'u + ρD'(z - w)
    2. z-update: z = shrink(Dq + w, α/ρ)  [soft thresholding]
    3. w-update: w = w + Dq - z           [dual variable]
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, vstack
from scipy.sparse.linalg import spsolve, cg
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from forward_solver import (
    create_disk_mesh,
    solve_poisson_zero_neumann,
    get_boundary_values,
)


class TVRegularizedSolver:
    """
    Linear inverse solver with proper Total Variation regularization.
    
    Uses ADMM to solve:
        min_q ||Gq - u_measured||^2 + α * TV(q)
    """
    
    def __init__(self, mesh_resolution=0.08, radius=1.0):
        self.radius = radius
        self.mesh_resolution = mesh_resolution
        
        print("Creating mesh...")
        self.mesh, _, _ = create_disk_mesh(radius, mesh_resolution)
        
        # Get coordinates
        self.all_coords = self.mesh.geometry.x[:, :2]
        radii = np.sqrt(self.all_coords[:, 0]**2 + self.all_coords[:, 1]**2)
        
        # Interior nodes (source locations)
        self.interior_mask = radii < 0.95 * radius
        self.interior_coords = self.all_coords[self.interior_mask]
        self.n_interior = len(self.interior_coords)
        self.interior_indices = np.where(self.interior_mask)[0]
        
        # Boundary nodes (measurements)
        self.boundary_mask = radii > 0.9 * radius
        self.boundary_coords = self.all_coords[self.boundary_mask]
        self.n_boundary = len(self.boundary_coords)
        self.boundary_angles = np.arctan2(self.boundary_coords[:, 1], self.boundary_coords[:, 0])
        
        sort_idx = np.argsort(self.boundary_angles)
        self.boundary_angles = self.boundary_angles[sort_idx]
        
        print(f"Interior nodes: {self.n_interior}")
        print(f"Boundary nodes: {self.n_boundary}")
        
        # Build operators
        self.G = None  # Green's matrix
        self.D = None  # Discrete gradient (TV operator)
        self.edges = None
        
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
        
    def build_gradient_operator(self):
        """
        Build discrete gradient operator D for TV regularization.
        
        D is an (n_edges x n_interior) matrix where each row corresponds
        to an edge (i,j) and has +1 at position i, -1 at position j.
        
        TV(q) = ||Dq||_1 = Σ_edges |q_i - q_j|
        """
        print("\nBuilding discrete gradient operator...")
        
        # Use Delaunay triangulation to find edges between interior nodes
        tri = Delaunay(self.interior_coords)
        
        # Extract unique edges from triangulation
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges_set.add(edge)
        
        self.edges = list(edges_set)
        n_edges = len(self.edges)
        
        print(f"Number of edges: {n_edges}")
        
        # Build sparse gradient matrix D
        # D[e, i] = +1, D[e, j] = -1 for edge e = (i, j)
        rows = []
        cols = []
        data = []
        
        for e, (i, j) in enumerate(self.edges):
            rows.extend([e, e])
            cols.extend([i, j])
            data.extend([1.0, -1.0])
        
        self.D = csr_matrix((data, (rows, cols)), shape=(n_edges, self.n_interior))
        
        # Compute edge lengths for weighted TV (optional, more accurate)
        self.edge_weights = np.array([
            np.linalg.norm(self.interior_coords[i] - self.interior_coords[j])
            for i, j in self.edges
        ])
        
        print("Gradient operator built!")
        
    def soft_threshold(self, x, threshold):
        """
        Soft thresholding (proximal operator for L1 norm).
        shrink(x, t) = sign(x) * max(|x| - t, 0)
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def solve_tv_admm(self, u_measured, alpha=1e-3, rho=1.0, max_iter=200, 
                      tol=1e-6, verbose=True, live_plot=False, sources_true=None):
        """
        Solve TV-regularized inverse problem using ADMM.
        
        min_q ||Gq - u||^2 + α * ||Dq||_1
        
        Parameters
        ----------
        u_measured : np.ndarray
            Measured boundary data
        alpha : float
            TV regularization parameter
        rho : float
            ADMM penalty parameter
        max_iter : int
            Maximum ADMM iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print iteration info
        live_plot : bool
            Show live updates
        sources_true : list
            True sources for comparison in plots
            
        Returns
        -------
        q : np.ndarray
            Recovered source intensities
        info : dict
            Convergence information
        """
        if self.G is None:
            self.build_greens_matrix()
        if self.D is None:
            self.build_gradient_operator()
        
        n = self.n_interior
        n_edges = self.D.shape[0]
        
        G = self.G
        D = self.D
        
        # Precompute matrices for q-update
        # (G'G + ρD'D) q = G'u + ρD'(z - w)
        GtG = G.T @ G
        DtD = (D.T @ D).toarray()  # Convert to dense for direct solve
        A = GtG + rho * DtD
        Gtu = G.T @ u_measured
        
        # Initialize variables
        q = np.zeros(n)
        z = np.zeros(n_edges)
        w = np.zeros(n_edges)  # Scaled dual variable
        
        # History for convergence
        history = {
            'primal_residual': [],
            'dual_residual': [],
            'objective': [],
            'tv_norm': [],
            'data_fidelity': []
        }
        
        if live_plot:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
        if verbose:
            print(f"\nSolving TV-regularized problem (α={alpha}, ρ={rho})...")
            print("-" * 60)
            print(f"{'Iter':>5} {'Primal':>12} {'Dual':>12} {'Objective':>12} {'TV':>12}")
            print("-" * 60)
        
        for k in range(max_iter):
            q_old = q.copy()
            
            # ========== q-update ==========
            # Solve (G'G + ρD'D)q = G'u + ρD'(z - w)
            rhs = Gtu + rho * D.T @ (z - w)
            q = np.linalg.solve(A, rhs)
            
            # ========== z-update ==========
            # z = shrink(Dq + w, α/ρ)
            Dq = D @ q
            z = self.soft_threshold(Dq + w, alpha / rho)
            
            # ========== w-update (dual) ==========
            w = w + Dq - z
            
            # ========== Convergence checks ==========
            # Primal residual: ||Dq - z||
            primal_res = np.linalg.norm(Dq - z)
            
            # Dual residual: ρ||D'(z - z_old)||
            # Approximate with change in q
            dual_res = rho * np.linalg.norm(D.T @ (z - (D @ q_old + w - (w - Dq + z))))
            dual_res = np.linalg.norm(q - q_old)  # Simpler approximation
            
            # Objective value
            data_fit = np.linalg.norm(G @ q - u_measured)**2
            tv_norm = np.sum(np.abs(Dq))
            objective = data_fit + alpha * tv_norm
            
            history['primal_residual'].append(primal_res)
            history['dual_residual'].append(dual_res)
            history['objective'].append(objective)
            history['tv_norm'].append(tv_norm)
            history['data_fidelity'].append(data_fit)
            
            if verbose and k % 10 == 0:
                print(f"{k:5d} {primal_res:12.4e} {dual_res:12.4e} {objective:12.4e} {tv_norm:12.4e}")
            
            # Live plotting
            if live_plot and k % 5 == 0:
                self._update_live_plot(fig, axes, q, u_measured, history, sources_true, k)
            
            # Check convergence
            if primal_res < tol and dual_res < tol:
                if verbose:
                    print(f"\nConverged at iteration {k}")
                break
        
        if verbose:
            print("-" * 60)
            print(f"Final: objective={objective:.4e}, TV={tv_norm:.4e}, data_fit={data_fit:.4e}")
        
        if live_plot:
            plt.ioff()
            plt.show()
        
        # Enforce zero mean
        q = q - np.mean(q)
        
        return q, history
    
    def solve_tv_primal_dual(self, u_measured, alpha=1e-3, tau=0.1, sigma=0.1,
                             max_iter=500, tol=1e-6, verbose=True):
        """
        Solve TV-regularized problem using Chambolle-Pock primal-dual algorithm.
        
        This is an alternative to ADMM, often faster for TV problems.
        
        min_q ||Gq - u||^2 + α * ||Dq||_1
        
        Primal-dual form:
        min_q max_p  ||Gq - u||^2 + <Dq, p> - δ_{||p||_∞ ≤ α}(p)
        """
        if self.G is None:
            self.build_greens_matrix()
        if self.D is None:
            self.build_gradient_operator()
        
        n = self.n_interior
        n_edges = self.D.shape[0]
        
        G = self.G
        D = self.D
        
        # Initialize
        q = np.zeros(n)
        q_bar = np.zeros(n)
        p = np.zeros(n_edges)
        
        # Precompute
        GtG = G.T @ G
        Gtu = G.T @ u_measured
        
        history = {'objective': [], 'tv_norm': []}
        
        if verbose:
            print(f"\nSolving TV problem with Chambolle-Pock (α={alpha})...")
        
        for k in range(max_iter):
            # Dual update: p = proj_{||.||_∞ ≤ α}(p + σ * D @ q_bar)
            p = p + sigma * (D @ q_bar)
            # Project onto ||p||_∞ ≤ α
            p = np.clip(p, -alpha, alpha)
            
            # Primal update
            q_old = q.copy()
            # q = (I + τ G'G)^{-1} (q - τ D'p + τ G'u)
            rhs = q - tau * (D.T @ p) + tau * Gtu
            # Solve (I + τ G'G) q_new = rhs
            q = np.linalg.solve(np.eye(n) + tau * GtG, rhs)
            
            # Extrapolation
            q_bar = 2 * q - q_old
            
            # Check convergence
            change = np.linalg.norm(q - q_old) / (np.linalg.norm(q_old) + 1e-10)
            
            Dq = D @ q
            objective = np.linalg.norm(G @ q - u_measured)**2 + alpha * np.sum(np.abs(Dq))
            tv_norm = np.sum(np.abs(Dq))
            
            history['objective'].append(objective)
            history['tv_norm'].append(tv_norm)
            
            if verbose and k % 50 == 0:
                print(f"  Iter {k}: obj={objective:.4e}, TV={tv_norm:.4e}, change={change:.4e}")
            
            if change < tol:
                if verbose:
                    print(f"Converged at iteration {k}")
                break
        
        q = q - np.mean(q)
        return q, history
    
    def _update_live_plot(self, fig, axes, q, u_measured, history, sources_true, iteration):
        """Update live plot during optimization."""
        for ax in axes.flat:
            ax.clear()
        
        # Plot 1: Source distribution
        ax = axes[0, 0]
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        
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
        
        ax.set_title(f'Source Distribution (iter {iteration})')
        ax.set_aspect('equal')
        
        # Plot 2: Boundary fit
        ax = axes[0, 1]
        ax.plot(self.boundary_angles, u_measured, 'b-', linewidth=2, label='Measured')
        ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Reconstructed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Boundary Fit')
        
        # Plot 3: Objective
        ax = axes[1, 0]
        if history['objective']:
            ax.semilogy(history['objective'], 'b-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.set_title('Convergence')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: TV norm
        ax = axes[1, 1]
        if history['tv_norm']:
            ax.plot(history['tv_norm'], 'g-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('TV(q)')
        ax.set_title('Total Variation')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'TV Regularization - ADMM Iteration {iteration}')
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    
    def compare_regularizations(self, u_measured, sources_true=None, save_path=None):
        """
        Compare L1, L2, and TV regularization side by side.
        """
        if self.G is None:
            self.build_greens_matrix()
        if self.D is None:
            self.build_gradient_operator()
        
        print("\n" + "="*60)
        print("COMPARING REGULARIZATION METHODS")
        print("="*60)
        
        # L2 (Tikhonov)
        print("\n1. L2 (Tikhonov)...")
        alpha_l2 = 1e-4
        GtG = self.G.T @ self.G
        Gtu = self.G.T @ u_measured
        q_l2 = np.linalg.solve(GtG + alpha_l2 * np.eye(self.n_interior), Gtu)
        q_l2 = q_l2 - np.mean(q_l2)
        
        # L1 (Sparsity)
        print("2. L1 (Sparsity)...")
        q_l1 = self._solve_l1(u_measured, alpha=1e-4)
        
        # TV
        print("3. TV (Total Variation)...")
        q_tv, _ = self.solve_tv_admm(u_measured, alpha=1e-3, rho=1.0, 
                                      max_iter=200, verbose=False)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        methods = [('L2 (Tikhonov)', q_l2), ('L1 (Sparsity)', q_l1), ('TV', q_tv)]
        
        for col, (name, q) in enumerate(methods):
            # Top row: source distribution
            ax = axes[0, col]
            coords = self.mesh.geometry.x
            self.mesh.topology.create_connectivity(2, 0)
            cells = self.mesh.topology.connectivity(2, 0)
            triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
            tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
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
            
            # Bottom row: boundary fit
            ax = axes[1, col]
            ax.plot(self.boundary_angles, u_measured, 'b-', linewidth=2, label='Measured')
            ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Fit')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            residual = np.linalg.norm(self.G @ q - u_measured)
            ax.set_title(f'Boundary Fit (residual={residual:.4f})')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return {'l2': q_l2, 'l1': q_l1, 'tv': q_tv}
    
    def _solve_l1(self, u_measured, alpha=1e-4, max_iter=50):
        """L1 via IRLS for comparison."""
        q = np.zeros(self.n_interior)
        epsilon = 1e-4
        
        for _ in range(max_iter):
            W = np.diag(1.0 / (np.abs(q) + epsilon))
            GtG = self.G.T @ self.G
            Gtu = self.G.T @ u_measured
            q_new = np.linalg.solve(GtG + alpha * W, Gtu)
            
            if np.linalg.norm(q_new - q) < 1e-6:
                break
            q = q_new
        
        return q - np.mean(q)
    
    def plot_results(self, q, sources_true=None, title='TV Regularization', save_path=None):
        """Plot final results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Source distribution
        ax = axes[0]
        coords = self.mesh.geometry.x
        self.mesh.topology.create_connectivity(2, 0)
        cells = self.mesh.topology.connectivity(2, 0)
        triangles = np.array([cells.links(i) for i in range(cells.num_nodes)])
        tri = Triangulation(coords[:, 0], coords[:, 1], triangles)
        ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.3)
        
        vmax = max(np.max(np.abs(q)), 0.01)
        scatter = ax.scatter(
            self.interior_coords[:, 0], self.interior_coords[:, 1],
            c=q, cmap='RdBu_r', s=40, vmin=-vmax, vmax=vmax
        )
        plt.colorbar(scatter, ax=ax, label='Source intensity q')
        
        if sources_true:
            for (x, y), intensity in sources_true:
                marker = 'o' if intensity > 0 else 's'
                ax.plot(x, y, marker, color='black', markersize=15,
                       markerfacecolor='none', markeredgewidth=3)
        
        ax.set_title(f'{title} - Source Distribution')
        ax.set_aspect('equal')
        
        # Boundary fit
        ax = axes[1]
        if hasattr(self, 'u_measured'):
            ax.plot(self.boundary_angles, self.u_measured, 'b-', linewidth=2, label='Measured')
        ax.plot(self.boundary_angles, self.G @ q, 'r--', linewidth=2, label='Reconstructed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Boundary Fit')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Value')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# MAIN
# ============================================================================

def generate_synthetic_data(sources, noise_level=0.0, mesh_resolution=0.05):
    """Generate synthetic boundary measurements."""
    mesh, _, _ = create_disk_mesh(resolution=mesh_resolution)
    u = solve_poisson_zero_neumann(mesh, sources)
    angles, values = get_boundary_values(u)
    if noise_level > 0:
        values = values + np.random.normal(0, noise_level, len(values))
    return angles, values


if __name__ == "__main__":
    print("="*60)
    print("Total Variation Regularization Demo")
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
    
    # Create solver
    solver = TVRegularizedSolver(mesh_resolution=0.08)
    solver.build_greens_matrix()
    solver.build_gradient_operator()
    
    # Interpolate to solver's boundary angles
    interp = interp1d(
        np.concatenate([angles - 2*np.pi, angles, angles + 2*np.pi]),
        np.tile(values, 3), kind='linear'
    )
    u_measured = interp(solver.boundary_angles)
    solver.u_measured = u_measured
    
    # Solve with TV (ADMM)
    print("\n" + "="*60)
    print("TV REGULARIZATION (ADMM)")
    print("="*60)
    q_tv, history = solver.solve_tv_admm(
        u_measured, 
        alpha=1e-3,      # TV weight
        rho=1.0,         # ADMM penalty
        max_iter=200,
        verbose=True,
        live_plot=True,
        sources_true=sources_true
    )
    
    solver.plot_results(q_tv, sources_true, title='TV (ADMM)', 
                       save_path='results/inverse_tv_admm.png')
    
    # Compare all methods
    print("\n" + "="*60)
    print("COMPARING ALL REGULARIZATION METHODS")
    print("="*60)
    results = solver.compare_regularizations(
        u_measured, 
        sources_true=sources_true,
        save_path='results/regularization_comparison.png'
    )
    
    print("\nDone!")
