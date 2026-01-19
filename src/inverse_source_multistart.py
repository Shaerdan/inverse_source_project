"""
INVERSE SOURCE LOCALIZATION - with multistart for robustness

Key fixes:
1. Push initial point to interior (mimics MATLAB)
2. Multistart with different initializations to avoid bad local minima
"""

import numpy as np
from scipy.optimize import minimize
import time


def forward_model(theta, x, n_sources):
    """Forward model - same as MATLAB"""
    phi = np.zeros_like(theta)
    for i in range(n_sources):
        S = x[3*i]
        rs = x[3*i + 1]
        ths = x[3*i + 2]
        arg = 1 + rs**2 - 2*rs*np.cos(theta - ths)
        arg = np.maximum(arg, 1e-30)
        phi += (S / (2*np.pi)) * np.log(np.sqrt(arg))
    return 2 * phi


def objective(x, theta, data, n_sources):
    phi = forward_model(theta, x, n_sources)
    return np.sum((phi - data)**2)


def gradient(x, theta, data, n_sources):
    phi = forward_model(theta, x, n_sources)
    residual = phi - data
    
    grad = np.zeros(3*n_sources)
    for i in range(n_sources):
        S = x[3*i]
        rs = x[3*i + 1]
        ths = x[3*i + 2]
        
        arg = 1 + rs**2 - 2*rs*np.cos(theta - ths)
        arg = np.maximum(arg, 1e-30)
        
        dphi_dS = (1/(2*np.pi)) * np.log(arg)
        dphi_drs = (S/np.pi) * (rs - np.cos(theta - ths)) / arg
        dphi_dths = -(S*rs/np.pi) * np.sin(theta - ths) / arg
        
        grad[3*i] = 2 * np.sum(residual * dphi_dS)
        grad[3*i + 1] = 2 * np.sum(residual * dphi_drs)
        grad[3*i + 2] = 2 * np.sum(residual * dphi_dths)
    
    return grad


def create_sources(n_sources, r_value=0.8, seed=42):
    """Create test sources - same as MATLAB"""
    np.random.seed(seed)
    ths = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    ths = ths + 0.1 * (2*np.random.rand(n_sources) - 1)
    S = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])
    S[-1] -= np.sum(S)
    rs = r_value * np.ones(n_sources)
    
    x = np.zeros(3*n_sources)
    for i in range(n_sources):
        x[3*i] = S[i]
        x[3*i + 1] = rs[i]
        x[3*i + 2] = ths[i]
    return x


def generate_initial_point(n_sources, K, method='linspace', seed=None):
    """
    Generate initial point using different strategies.
    
    Methods:
    - 'linspace': MATLAB-style linspace(-4, 4) - deterministic
    - 'spread': Sources spread evenly around circle - deterministic  
    - 'random': Random initialization - uses seed
    """
    if method == 'linspace':
        x0 = np.linspace(-4, 4, 3 * n_sources)
    
    elif method == 'spread':
        # Sources evenly spread around circle
        x0 = np.zeros(3 * n_sources)
        angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
        for i in range(n_sources):
            x0[3*i] = 0.5 * (1 if i % 2 == 0 else -1)  # Alternating
            x0[3*i + 1] = 0.5  # r = 0.5
            x0[3*i + 2] = angles[i]
        # Fix sum constraint
        x0[3*(n_sources-1)] = -sum(x0[3*i] for i in range(n_sources-1))
    
    elif method == 'random':
        if seed is not None:
            np.random.seed(seed)
        x0 = np.zeros(3 * n_sources)
        for i in range(n_sources):
            x0[3*i] = np.random.uniform(-K*0.8, K*0.8)      # S
            x0[3*i + 1] = np.random.uniform(0.1, 0.9)       # r (positive)
            x0[3*i + 2] = np.random.uniform(-np.pi, np.pi)  # θ
        # Fix sum constraint
        x0[3*(n_sources-1)] = -sum(x0[3*i] for i in range(n_sources-1))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Push to interior
    return push_to_interior(x0, n_sources, K)


def push_to_interior(x0, n_sources, K, margin=0.1):
    """Push initial point away from bounds - mimics MATLAB's behavior."""
    x = x0.copy()
    for i in range(n_sources):
        x[3*i] = np.clip(x[3*i], -K * (1 - margin), K * (1 - margin))
        x[3*i + 1] = np.clip(x[3*i + 1], -(1 - margin), 1 - margin)
        x[3*i + 2] = np.clip(x[3*i + 2], -np.pi + margin, np.pi - margin)
    return x


def solve_single(n_sources, theta, data, K, x0):
    """Single optimization run from given initial point."""
    bounds = []
    for i in range(n_sources):
        bounds.append((-K, K))
        bounds.append((-1, 1))
        bounds.append((-np.pi, np.pi))
    
    constraints = {
        'type': 'eq',
        'fun': lambda x: sum(x[3*i] for i in range(n_sources)),
        'jac': lambda x: np.array([1.0 if i % 3 == 0 else 0.0 for i in range(3 * n_sources)])
    }
    
    result = minimize(
        fun=objective,
        x0=x0,
        args=(theta, data, n_sources),
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 30000, 'ftol': 1e-16, 'disp': False}
    )
    
    return result.x, result.fun, result


def solve_multistart(n_sources, theta, data, K, n_starts=5, verbose=True):
    """
    Solve with multistart - try multiple initial points, keep best result.
    """
    best_x = None
    best_obj = np.inf
    best_result = None
    
    # Generate different initial points
    init_methods = ['linspace', 'spread'] + [('random', i) for i in range(n_starts - 2)]
    
    for i, method in enumerate(init_methods):
        if isinstance(method, tuple):
            x0 = generate_initial_point(n_sources, K, method='random', seed=method[1] * 100 + n_sources)
            method_name = f'random_{method[1]}'
        else:
            x0 = generate_initial_point(n_sources, K, method=method)
            method_name = method
        
        x_sol, obj, result = solve_single(n_sources, theta, data, K, x0)
        
        if verbose:
            print(f"    Start {i+1} ({method_name}): obj = {obj:.2e}")
        
        if obj < best_obj:
            best_obj = obj
            best_x = x_sol
            best_result = result
    
    return best_x, best_result


def compute_errors(x_true, x_sol, n_sources):
    """Compute position and intensity errors with greedy matching"""
    true_pts = []
    rec_pts = []
    for i in range(n_sources):
        r_t, th_t, S_t = x_true[3*i+1], x_true[3*i+2], x_true[3*i]
        r_r, th_r, S_r = x_sol[3*i+1], x_sol[3*i+2], x_sol[3*i]
        true_pts.append((r_t*np.cos(th_t), r_t*np.sin(th_t), S_t))
        rec_pts.append((r_r*np.cos(th_r), r_r*np.sin(th_r), S_r))
    
    used = set()
    pos_err = []
    int_err = []
    for i, (xt, yt, St) in enumerate(true_pts):
        best_d, best_j = np.inf, -1
        for j, (xr, yr, Sr) in enumerate(rec_pts):
            if j in used:
                continue
            d = np.sqrt((xt-xr)**2 + (yt-yr)**2)
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        pos_err.append(best_d)
        int_err.append(abs(St - rec_pts[best_j][2]))
    
    return np.mean(pos_err), np.max(pos_err), np.mean(int_err), np.max(int_err)


def test(n_sources, n_starts=5, verbose=True):
    """Test inverse problem for given number of sources"""
    print(f"\n{'='*60}")
    print(f"Testing {n_sources} sources (multistart with {n_starts} attempts)")
    print('='*60)
    
    theta = np.linspace(-np.pi, np.pi, 1000)
    x_true = create_sources(n_sources, seed=42)
    data = forward_model(theta, x_true, n_sources)
    
    K = max(2.0, 2 * np.max(np.abs(x_true[::3])))
    
    if verbose:
        print(f"\nTrue sources:")
        for i in range(n_sources):
            print(f"  {i+1}: S={x_true[3*i]:+.4f}, r={x_true[3*i+1]:.4f}, θ={x_true[3*i+2]:.4f}")
        print(f"\nMultistart progress:")
    
    t0 = time.time()
    x_sol, result = solve_multistart(n_sources, theta, data, K, n_starts=n_starts, verbose=verbose)
    elapsed = time.time() - t0
    
    obj = objective(x_sol, theta, data, n_sources)
    pos_mean, pos_max, int_mean, int_max = compute_errors(x_true, x_sol, n_sources)
    
    print(f"\nBest result:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Objective: {obj:.2e}")
    print(f"  Position error: mean={pos_mean:.2e}, max={pos_max:.2e}")
    print(f"  Intensity error: mean={int_mean:.2e}, max={int_max:.2e}")
    
    if verbose:
        print(f"\nRecovered sources:")
        for i in range(n_sources):
            print(f"  {i+1}: S={x_sol[3*i]:+.4f}, r={x_sol[3*i+1]:.4f}, θ={x_sol[3*i+2]:.4f}")
    
    return {'n': n_sources, 'obj': obj, 'pos_mean': pos_mean, 'pos_max': pos_max, 
            'time': elapsed}


if __name__ == '__main__':
    import sys
    
    verbose = '-v' in sys.argv
    n_starts = 5  # Number of multistart attempts
    
    print("="*60)
    print("INVERSE SOURCE LOCALIZATION - MULTISTART VERSION")
    print("="*60)
    print(f"\nUsing {n_starts} random starts to avoid bad local minima")
    
    results = []
    for n in [6, 8, 10, 12]:
        r = test(n, n_starts=n_starts, verbose=verbose)
        results.append(r)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'N':>4} | {'Objective':>12} | {'Pos Mean':>12} | {'Pos Max':>12} | {'Time':>6}")
    print("-"*60)
    for r in results:
        print(f"{r['n']:4d} | {r['obj']:12.2e} | {r['pos_mean']:12.2e} | {r['pos_max']:12.2e} | {r['time']:5.1f}s")
    
    print("\nMATLAB reference: 12 sources → Objective ~2e-10, Position ~1e-5")
