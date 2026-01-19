"""
MATLAB fmincon reproduction test.

Polar parameterization: [S1, r1, θ1, S2, r2, θ2, ...]
Box bounds: S ∈ [-K, K], r ∈ [-1, 1], θ ∈ [-π, π]
Equality constraint: sum(S) = 0

Test: 6, 8, 10, 12 sources, all at r=0.8
"""

import numpy as np
from scipy.optimize import minimize
import time


def forward_polar(theta_boundary, params, n_sources):
    """
    MATLAB:
        phi(:,i) = (S(i)/(2*pi))*log(sqrt(1+r(i)^2 - 2*r(i)*cos(theta-ths(i))));
        phi = 2*sum(phi,2);
    """
    phi = np.zeros_like(theta_boundary)
    for i in range(n_sources):
        S = params[3*i]
        r = params[3*i + 1]
        th = params[3*i + 2]
        arg = 1 + r**2 - 2*r*np.cos(theta_boundary - th)
        arg = np.maximum(arg, 1e-14)
        phi += (S / (2*np.pi)) * np.log(np.sqrt(arg))
    return 2 * phi


def objective(params, n_sources, theta_boundary, u_measured):
    """Least squares: ||u_computed - u_measured||²"""
    u = forward_polar(theta_boundary, params, n_sources)
    return np.sum((u - u_measured)**2)


def sum_constraint(params, n_sources):
    """Sum of intensities = 0"""
    return sum(params[3*i] for i in range(n_sources))


def create_sources(n_sources, r=0.8, seed=None):
    """Create test sources at fixed radius, evenly spaced angles, intensities sum to 0."""
    if seed is not None:
        np.random.seed(seed)
    
    # Evenly spaced angles with small random perturbation
    angles = np.linspace(-np.pi, np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)  # Small perturbation
    
    # Random intensities that sum to 0 (like MATLAB randfixedsum)
    intensities = np.random.randn(n_sources)
    intensities = intensities - np.mean(intensities)  # Force sum = 0
    
    # Build params array [S1, r1, θ1, S2, r2, θ2, ...]
    params = []
    for i in range(n_sources):
        params.extend([intensities[i], r, angles[i]])
    
    return np.array(params)


def solve_inverse(u_measured, n_sources, theta_boundary, n_restarts=30, verbose=False):
    """
    Solve inverse problem using SLSQP with multiple restarts.
    """
    K = 5.0  # Intensity bound
    
    # Bounds: [S, r, θ] for each source
    bounds = []
    for _ in range(n_sources):
        bounds.append((-K, K))           # S
        bounds.append((-1.0, 1.0))       # r (THIS enforces disk constraint)
        bounds.append((-np.pi, np.pi))   # θ
    
    # Equality constraint: sum(S) = 0
    constraint = {'type': 'eq', 'fun': lambda p: sum_constraint(p, n_sources)}
    
    best_result = None
    best_fun = np.inf
    
    for restart in range(n_restarts):
        # Random initial guess
        x0 = []
        angles_init = np.sort(np.random.uniform(-np.pi, np.pi, n_sources))
        for i in range(n_sources):
            S_init = np.random.uniform(-1, 1)
            r_init = np.random.uniform(0.3, 0.9)
            x0.extend([S_init, r_init, angles_init[i]])
        x0 = np.array(x0)
        
        # Center intensities to satisfy constraint approximately
        S_sum = sum(x0[3*i] for i in range(n_sources))
        for i in range(n_sources):
            x0[3*i] -= S_sum / n_sources
        
        try:
            result = minimize(
                objective,
                x0,
                args=(n_sources, theta_boundary, u_measured),
                method='SLSQP',
                bounds=bounds,
                constraints=constraint,
                options={'maxiter': 10000, 'ftol': 1e-12}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
                
            if verbose:
                print(f"  Restart {restart+1}: obj = {result.fun:.2e}, success = {result.success}")
                
        except Exception as e:
            if verbose:
                print(f"  Restart {restart+1}: FAILED - {e}")
    
    return best_result


def extract_sources(params, n_sources):
    """Extract (x, y, S) from polar params."""
    sources = []
    for i in range(n_sources):
        S = params[3*i]
        r = params[3*i + 1]
        th = params[3*i + 2]
        x = r * np.cos(th)
        y = r * np.sin(th)
        sources.append((x, y, S))
    return sources


def compute_errors(true_params, recovered_params, n_sources):
    """Compute position and intensity errors using greedy matching."""
    true_sources = extract_sources(true_params, n_sources)
    rec_sources = extract_sources(recovered_params, n_sources)
    
    # Greedy matching: for each true source, find closest recovered
    used = set()
    pos_errors = []
    int_errors = []
    
    for (x_t, y_t, S_t) in true_sources:
        best_dist = np.inf
        best_idx = -1
        best_rec = None
        
        for idx, (x_r, y_r, S_r) in enumerate(rec_sources):
            if idx in used:
                continue
            dist = np.sqrt((x_t - x_r)**2 + (y_t - y_r)**2)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
                best_rec = (x_r, y_r, S_r)
        
        used.add(best_idx)
        pos_errors.append(best_dist)
        int_errors.append(abs(S_t - best_rec[2]))
    
    return np.mean(pos_errors), np.max(pos_errors), np.mean(int_errors), np.max(int_errors)


def test_n_sources(n_sources, seed=42):
    """Test recovery for given number of sources."""
    print(f"\n{'='*60}")
    print(f"Testing {n_sources} sources (r=0.8, evenly spaced angles)")
    print('='*60)
    
    # Setup
    n_boundary = 1000
    theta_boundary = np.linspace(-np.pi, np.pi, n_boundary)
    
    # Create ground truth
    true_params = create_sources(n_sources, r=0.8, seed=seed)
    
    # Generate measurements (no noise for now)
    u_measured = forward_polar(theta_boundary, true_params, n_sources)
    
    # Print true sources
    print("\nTrue sources:")
    true_sources = extract_sources(true_params, n_sources)
    for i, (x, y, S) in enumerate(true_sources):
        r = np.sqrt(x**2 + y**2)
        th = np.arctan2(y, x)
        print(f"  {i+1}: r={r:.4f}, θ={th:+.4f}, S={S:+.4f}")
    print(f"  Sum of intensities: {sum(s[2] for s in true_sources):.2e}")
    
    # Solve
    print("\nSolving inverse problem...")
    t0 = time.time()
    result = solve_inverse(u_measured, n_sources, theta_boundary, n_restarts=30, verbose=False)
    elapsed = time.time() - t0
    
    if result is None:
        print("FAILED: No solution found")
        return None
    
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final objective: {result.fun:.2e}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    
    # Print recovered sources
    print("\nRecovered sources:")
    rec_sources = extract_sources(result.x, n_sources)
    for i, (x, y, S) in enumerate(rec_sources):
        r = np.sqrt(x**2 + y**2)
        th = np.arctan2(y, x)
        print(f"  {i+1}: r={r:.4f}, θ={th:+.4f}, S={S:+.4f}")
    print(f"  Sum of intensities: {sum(s[2] for s in rec_sources):.2e}")
    
    # Compute errors
    mean_pos, max_pos, mean_int, max_int = compute_errors(true_params, result.x, n_sources)
    print(f"\nErrors:")
    print(f"  Position: mean={mean_pos:.2e}, max={max_pos:.2e}")
    print(f"  Intensity: mean={mean_int:.2e}, max={max_int:.2e}")
    
    return {
        'n_sources': n_sources,
        'objective': result.fun,
        'success': result.success,
        'mean_pos_error': mean_pos,
        'max_pos_error': max_pos,
        'mean_int_error': mean_int,
        'max_int_error': max_int,
        'time': elapsed
    }


if __name__ == '__main__':
    print("MATLAB fmincon Reproduction Test")
    print("Polar parameterization, SLSQP optimizer")
    print("Ground truth: all sources at r=0.8")
    
    results = []
    for n in [6, 8, 10, 12]:
        r = test_n_sources(n, seed=42)
        if r:
            results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'N':>4} | {'Objective':>12} | {'Pos Error':>12} | {'Int Error':>12} | {'Time':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['n_sources']:4d} | {r['objective']:12.2e} | {r['mean_pos_error']:12.2e} | {r['mean_int_error']:12.2e} | {r['time']:7.1f}s")
