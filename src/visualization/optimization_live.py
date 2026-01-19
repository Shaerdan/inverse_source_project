"""
Real-Time Optimization Visualization
====================================

Provides live visualization during IPOPT optimization, similar to MATLAB's fmincon.

Features:
- Loss curve updating in real-time
- Bar chart of current source intensities
- 2D plot showing current source positions vs true positions
- Convergence diagnostics

Usage:
    solver = IPOPTNonlinearInverseSolver(n_sources=4, n_boundary=100)
    solver.set_measured_data(u_measured)
    result = solver.solve_with_visualization(sources_true=sources_true)

Or standalone:
    viz = OptimizationVisualizer(n_sources=4, sources_true=sources_true)
    # ... during optimization callback:
    viz.update(iteration, objective, x_current)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional
import time


class OptimizationVisualizer:
    """
    Real-time visualization of optimization progress.
    
    Creates a figure with:
    - Top-left: Loss curve (log scale)
    - Top-right: Current source positions (2D scatter)
    - Bottom-left: Intensity bar chart
    - Bottom-right: Convergence info text
    
    Parameters
    ----------
    n_sources : int
        Number of sources being optimized
    sources_true : list, optional
        True source positions for comparison: [((x,y), q), ...]
    domain : str
        Domain type for boundary drawing ('disk', 'ellipse', etc.)
    domain_params : dict
        Parameters for domain (e.g., {'a': 1.5, 'b': 1.0} for ellipse)
    update_interval : int
        Update plot every N iterations (default: 1)
    figsize : tuple
        Figure size (default: (12, 8))
    """
    
    def __init__(self, n_sources: int, sources_true: List[Tuple] = None,
                 domain: str = 'disk', domain_params: dict = None,
                 update_interval: int = 1, figsize: tuple = (12, 8)):
        self.n_sources = n_sources
        self.sources_true = sources_true
        self.domain = domain
        self.domain_params = domain_params or {}
        self.update_interval = update_interval
        
        # History storage
        self.iterations = []
        self.objectives = []
        self.positions_history = []
        self.intensities_history = []
        
        # Timing
        self.start_time = None
        self.iteration_times = []
        
        # Setup interactive plotting
        plt.ion()
        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(2, 2, figure=self.fig, hspace=0.3, wspace=0.3)
        
        self._setup_axes()
        self._draw_initial()
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def _setup_axes(self):
        """Setup the four subplot axes."""
        # Loss curve (top-left)
        self.ax_loss = self.fig.add_subplot(self.gs[0, 0])
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Objective (log scale)')
        self.ax_loss.set_title('Loss Curve')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Source positions (top-right)
        self.ax_pos = self.fig.add_subplot(self.gs[0, 1])
        self.ax_pos.set_xlabel('x')
        self.ax_pos.set_ylabel('y')
        self.ax_pos.set_title('Source Positions')
        self.ax_pos.set_aspect('equal')
        self.ax_pos.grid(True, alpha=0.3)
        
        # Intensity bars (bottom-left)
        self.ax_bars = self.fig.add_subplot(self.gs[1, 0])
        self.ax_bars.set_xlabel('Source Index')
        self.ax_bars.set_ylabel('Intensity')
        self.ax_bars.set_title('Source Intensities')
        self.ax_bars.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Info text (bottom-right)
        self.ax_info = self.fig.add_subplot(self.gs[1, 1])
        self.ax_info.axis('off')
        self.ax_info.set_title('Convergence Info')
    
    def _draw_initial(self):
        """Draw initial state (domain boundary, true sources)."""
        # Draw domain boundary
        self._draw_domain_boundary()
        
        # Draw true sources if provided
        if self.sources_true is not None:
            for i, ((x, y), q) in enumerate(self.sources_true):
                color = 'green' if q > 0 else 'red'
                self.ax_pos.scatter([x], [y], c=color, marker='*', s=200, 
                                   edgecolors='black', linewidths=1, 
                                   label='True' if i == 0 else '', zorder=10)
            
            # True intensities as reference lines on bar chart
            true_q = [s[1] for s in self.sources_true]
            for i, q in enumerate(true_q):
                self.ax_bars.axhline(y=q, color='gray', linestyle='--', 
                                    alpha=0.5, xmin=(i+0.1)/self.n_sources, 
                                    xmax=(i+0.9)/self.n_sources)
        
        # Initialize empty plots for current state
        self.loss_line, = self.ax_loss.plot([], [], 'b-', linewidth=1.5)
        self.loss_point, = self.ax_loss.plot([], [], 'bo', markersize=8)
        
        self.current_pos_scatter = None
        self.bars = None
        self.info_text = None
    
    def _draw_domain_boundary(self):
        """Draw the domain boundary."""
        theta = np.linspace(0, 2*np.pi, 200)
        
        if self.domain == 'disk':
            r = self.domain_params.get('r', 1.0)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        elif self.domain == 'ellipse':
            a = self.domain_params.get('a', 1.5)
            b = self.domain_params.get('b', 1.0)
            x = a * np.cos(theta)
            y = b * np.sin(theta)
        else:
            # Default: unit disk
            x = np.cos(theta)
            y = np.sin(theta)
        
        self.ax_pos.plot(x, y, 'k-', linewidth=2, label='Boundary')
        
        # Set axis limits with margin
        margin = 0.2
        self.ax_pos.set_xlim(x.min() - margin, x.max() + margin)
        self.ax_pos.set_ylim(y.min() - margin, y.max() + margin)
    
    def update(self, iteration: int, objective: float, x: np.ndarray):
        """
        Update visualization with current optimization state.
        
        Parameters
        ----------
        iteration : int
            Current iteration number
        objective : float
            Current objective value
        x : np.ndarray
            Current parameter vector [x1,y1,x2,y2,...,q1,q2,...]
        """
        if iteration % self.update_interval != 0:
            return
        
        # Record timing
        current_time = time.time()
        if self.start_time is None:
            self.start_time = current_time
        elapsed = current_time - self.start_time
        
        # Store history
        self.iterations.append(iteration)
        self.objectives.append(objective)
        
        # Extract positions and intensities
        n = self.n_sources
        positions = [(x[2*i], x[2*i+1]) for i in range(n)]
        intensities = np.array([x[2*n + i] for i in range(n)])
        intensities = intensities - np.mean(intensities)  # Center
        
        self.positions_history.append(positions)
        self.intensities_history.append(intensities.copy())
        
        # Update loss curve
        self.loss_line.set_data(self.iterations, self.objectives)
        self.loss_point.set_data([iteration], [objective])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # Update source positions
        if self.current_pos_scatter is not None:
            self.current_pos_scatter.remove()
        
        pos_x = [p[0] for p in positions]
        pos_y = [p[1] for p in positions]
        colors = ['blue' if q > 0 else 'orange' for q in intensities]
        sizes = 100 * np.abs(intensities) / max(np.abs(intensities).max(), 0.1)
        sizes = np.clip(sizes, 50, 300)
        
        self.current_pos_scatter = self.ax_pos.scatter(
            pos_x, pos_y, c=colors, s=sizes, marker='o',
            edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5
        )
        
        # Update intensity bars
        self.ax_bars.clear()
        self.ax_bars.set_xlabel('Source Index')
        self.ax_bars.set_ylabel('Intensity')
        self.ax_bars.set_title('Source Intensities')
        self.ax_bars.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        bar_colors = ['blue' if q > 0 else 'orange' for q in intensities]
        indices = np.arange(n)
        self.ax_bars.bar(indices, intensities, color=bar_colors, edgecolor='black', alpha=0.7)
        
        # Draw true intensities as markers if available
        if self.sources_true is not None:
            true_q = [s[1] for s in self.sources_true]
            self.ax_bars.scatter(indices, true_q, c='green', marker='*', s=150, 
                                zorder=10, label='True')
            self.ax_bars.legend(loc='upper right')
        
        self.ax_bars.set_xticks(indices)
        self.ax_bars.set_xticklabels([f'{i+1}' for i in range(n)])
        
        # Update info text
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_title('Convergence Info')
        
        # Compute position error if true sources known
        pos_error_str = ""
        if self.sources_true is not None:
            errors = []
            for (tx, ty), _ in self.sources_true:
                min_dist = min(np.sqrt((tx-px)**2 + (ty-py)**2) 
                              for px, py in positions)
                errors.append(min_dist)
            pos_error_str = f"Position Error (approx): {np.mean(errors):.2e}\n"
        
        # Convergence rate
        conv_rate_str = ""
        if len(self.objectives) > 10:
            recent = self.objectives[-10:]
            if recent[0] > 0:
                rate = (recent[-1] / recent[0]) ** (1/9)
                conv_rate_str = f"Convergence Rate: {rate:.4f}\n"
        
        info_text = (
            f"Iteration: {iteration}\n"
            f"Objective: {objective:.6e}\n"
            f"{pos_error_str}"
            f"Elapsed: {elapsed:.1f}s\n"
            f"{conv_rate_str}"
            f"\nSum(q): {intensities.sum():.2e}\n"
        )
        
        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def finalize(self, result=None):
        """
        Finalize visualization after optimization completes.
        
        Parameters
        ----------
        result : InverseResult, optional
            Final optimization result
        """
        if result is not None:
            # Draw final positions with different style
            final_x = [s.x for s in result.sources]
            final_y = [s.y for s in result.sources]
            self.ax_pos.scatter(final_x, final_y, c='purple', marker='D', s=150,
                               edgecolors='black', linewidths=2, label='Final', zorder=15)
            self.ax_pos.legend(loc='upper right')
        
        # Add title with final info
        if self.objectives:
            self.fig.suptitle(f'Optimization Complete - Final Objective: {self.objectives[-1]:.2e}',
                            fontsize=12, fontweight='bold')
        
        plt.ioff()
        plt.show()
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)


class IPOPTVisualizingProblem:
    """
    IPOPT problem wrapper that calls visualizer during optimization.
    
    Wraps an existing IPOPTDiskProblem or IPOPTConformalProblem and adds
    visualization callbacks.
    """
    
    def __init__(self, base_problem, visualizer: OptimizationVisualizer):
        self.base = base_problem
        self.viz = visualizer
        self.iter_count = 0
        
        # Copy attributes from base
        self.n_vars = base_problem.n_vars
        self.n_constraints = base_problem.n_constraints
        self.n_sources = base_problem.n_sources
    
    def objective(self, x):
        return self.base.objective(x)
    
    def gradient(self, x):
        return self.base.gradient(x)
    
    def constraints(self, x):
        return self.base.constraints(x)
    
    def jacobian(self, x):
        return self.base.jacobian(x)
    
    def jacobianstructure(self):
        return self.base.jacobianstructure()
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                    mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """
        IPOPT intermediate callback - called after each iteration.
        
        This is where we update the visualization.
        """
        self.iter_count = iter_count
        
        # Get current x from the base problem's last evaluation
        # Note: IPOPT doesn't pass x directly to intermediate, so we
        # use the objective value which was just computed
        if hasattr(self.base, '_last_x'):
            self.viz.update(iter_count, obj_value, self.base._last_x)
        
        return True  # Return True to continue optimization


def add_visualization_to_solver(solver_class):
    """
    Decorator/factory to add visualization capability to an IPOPT solver.
    
    This modifies the solver to optionally show real-time plots.
    """
    original_solve = solver_class.solve
    
    def solve_with_visualization(self, sources_true=None, show_viz=True, 
                                  update_interval=5, **kwargs):
        """
        Solve with optional real-time visualization.
        
        Parameters
        ----------
        sources_true : list, optional
            True sources for comparison: [((x,y), q), ...]
        show_viz : bool
            Whether to show visualization (default: True)
        update_interval : int
            Update plot every N iterations (default: 5)
        **kwargs : dict
            Additional arguments passed to solve()
        """
        if not show_viz:
            return original_solve(self, **kwargs)
        
        # Create visualizer
        viz = OptimizationVisualizer(
            n_sources=self.n_sources,
            sources_true=sources_true,
            update_interval=update_interval
        )
        
        # Store visualizer for callback access
        self._visualizer = viz
        
        # Run optimization
        result = original_solve(self, **kwargs)
        
        # Finalize visualization
        viz.finalize(result)
        
        return result
    
    solver_class.solve_with_visualization = solve_with_visualization
    return solver_class


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Real-Time Optimization Visualization")
    print("=" * 50)
    
    # Simple test without full IPOPT
    print("\nCreating test visualization...")
    
    n_sources = 4
    sources_true = [
        ((0.5, 0.3), 1.0),
        ((-0.4, 0.5), -0.8),
        ((-0.3, -0.4), 0.6),
        ((0.4, -0.3), -0.8)
    ]
    
    viz = OptimizationVisualizer(
        n_sources=n_sources,
        sources_true=sources_true,
        domain='disk',
        update_interval=1
    )
    
    # Simulate optimization iterations
    print("Simulating optimization progress...")
    np.random.seed(42)
    
    # Start with random positions
    x = np.random.randn(3 * n_sources) * 0.3
    
    for iteration in range(100):
        # Simulate objective decreasing
        objective = 10.0 * np.exp(-iteration / 20) + 0.001 * np.random.rand()
        
        # Simulate positions converging to true
        for i, ((tx, ty), tq) in enumerate(sources_true):
            alpha = 1 - np.exp(-iteration / 30)
            x[2*i] = (1 - alpha) * x[2*i] + alpha * tx + 0.01 * np.random.randn()
            x[2*i + 1] = (1 - alpha) * x[2*i + 1] + alpha * ty + 0.01 * np.random.randn()
            x[2*n_sources + i] = (1 - alpha) * x[2*n_sources + i] + alpha * tq + 0.01 * np.random.randn()
        
        viz.update(iteration, objective, x)
        time.sleep(0.05)  # Slow down for visibility
    
    print("\nSimulation complete!")
    viz.finalize()
    
    print("\nTo use with actual IPOPT solver:")
    print("  solver = IPOPTNonlinearInverseSolver(...)")
    print("  result = solver.solve_with_visualization(sources_true=sources_true)")
