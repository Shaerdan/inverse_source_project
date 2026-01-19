"""
FIXED IPOPT configuration matching MATLAB fmincon.

KEY FIX: Push initial point to interior before optimization.
MATLAB does this automatically, IPOPT does not.

Without this fix, linspace(-4,4) puts values at bounds, causing
gradient blow-up (|grad| ~ 10^16) and immediate termination.
"""

import numpy as np
import time

try:
    import cyipopt
    HAS_CYIPOPT = True
except ImportError:
    HAS_CYIPOPT = False
    print("cyipopt not available")


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


class InverseProblem:
    def __init__(self, n_sources, theta, data):
        self.n_sources = n_sources
        self.theta = theta
        self.data = data
        self.n = 3 * n_sources
        self.obj_count = 0
        self.grad_count = 0
    
    def objective(self, x):
        self.obj_count += 1
        phi = forward_model(self.theta, x, self.n_sources)
        return np.sum((phi - self.data)**2)
    
    def gradient(self, x):
        self.grad_count += 1
        n = self.n_sources
        theta = self.theta
        
        phi = forward_model(theta, x, n)
        residual = phi - self.data
        
        grad = np.zeros(3*n)
        for i in range(n):
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
    
    def constraints(self, x):
        return np.array([sum(x[3*i] for i in range(self.n_sources))])
    
    def jacobian(self, x):
        jac = np.zeros(self.n)
        for i in range(self.n_sources):
            jac[3*i] = 1.0
        return jac


def create_sources(n_sources, r_value=0.8, seed=42):
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


def push_to_interior(x0, n_sources, K, margin=0.1):
    """
    KEY FIX: Push initial point away from bounds.
    
    MATLAB doc: "fmincon resets x0 components that are on or outside 
    bounds lb or ub to values strictly between the bounds"
    
    Without this, linspace(-4,4) clipped to bounds puts many values
    exactly AT bounds, causing gradient blow-up (|grad| ~ 10^16).
    """
    x = x0.copy()
    for i in range(n_sources):
        # S: [-K, K] -> [-K*(1-margin), K*(1-margin)]
        s_lb, s_ub = -K * (1 - margin), K * (1 - margin)
        x[3*i] = np.clip(x[3*i], s_lb, s_ub)
        
        # r: [-1, 1] -> [-(1-margin), (1-margin)]
        x[3*i + 1] = np.clip(x[3*i + 1], -(1 - margin), 1 - margin)
        
        # θ: [-π, π] -> [-(π-margin), (π-margin)]
        x[3*i + 2] = np.clip(x[3*i + 2], -np.pi + margin, np.pi - margin)
    
    return x


def solve_ipopt_fixed(n_sources, theta, data, K, verbose=True):
    """
    IPOPT with fixed initialization.
    """
    if not HAS_CYIPOPT:
        raise RuntimeError("cyipopt not installed")
    
    # Initial guess - MATLAB style: linspace(-4, 4)
    x0_raw = np.linspace(-4, 4, 3*n_sources)
    
    # KEY FIX: Push to interior before optimization
    x0 = push_to_interior(x0_raw, n_sources, K, margin=0.1)
    
    # Bounds
    lb = np.array([-K, -1.0, -np.pi] * n_sources)
    ub = np.array([K, 1.0, np.pi] * n_sources)
    
    # Equality constraint: sum(S) = 0
    cl = np.array([0.0])
    cu = np.array([0.0])
    
    problem = InverseProblem(n_sources, theta, data)
    
    nlp = cyipopt.Problem(
        n=3*n_sources,
        m=1,
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    
    # IPOPT options - reasonable tolerances
    nlp.add_option('tol', 1e-12)
    nlp.add_option('dual_inf_tol', 1e-12)
    nlp.add_option('constr_viol_tol', 1e-14)
    nlp.add_option('compl_inf_tol', 1e-12)
    nlp.add_option('max_iter', 3000)
    
    # Barrier parameter
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('mu_init', 0.1)
    
    # Hessian approximation
    nlp.add_option('hessian_approximation', 'limited-memory')
    nlp.add_option('limited_memory_max_history', 20)
    
    # Output
    if verbose:
        nlp.add_option('print_level', 5)
        nlp.add_option('output_file', f'ipopt_fixed_n{n_sources}.txt')
    else:
        nlp.add_option('print_level', 0)
        nlp.add_option('sb', 'yes')
    
    # Solve
    x_sol, info = nlp.solve(x0)
    
    return x_sol, info, problem.obj_count, problem.grad_count


def compute_errors(x_true, x_sol, n_sources):
    true_pts = []
    rec_pts = []
    for i in range(n_sources):
        r_t, th_t, S_t = x_true[3*i+1], x_true[3*i+2], x_true[3*i]
        r_r, th_r, S_r = x_sol[3*i+1], x_sol[3*i+2], x_sol[3*i]
        true_pts.append((r_t*np.cos(th_t), r_t*np.sin(th_t), S_t))
        rec_pts.append((r_r*np.cos(th_r), r_r*np.sin(th_r), S_r))
    
    used = set()
    pos_err = []
    for (xt, yt, St) in true_pts:
        best_d, best_j = np.inf, -1
        for j, (xr, yr, Sr) in enumerate(rec_pts):
            if j in used:
                continue
            d = np.sqrt((xt-xr)**2 + (yt-yr)**2)
            if d < best_d:
                best_d, best_j = d, j
        used.add(best_j)
        pos_err.append(best_d)
    return np.mean(pos_err), np.max(pos_err)


def test(n_sources, verbose=False):
    print(f"\n{'='*60}")
    print(f"Testing {n_sources} sources")
    print('='*60)
    
    theta = np.linspace(-np.pi, np.pi, 1000)
    x_true = create_sources(n_sources, seed=42)
    data = forward_model(theta, x_true, n_sources)
    
    K = max(2.0, 2*np.max(np.abs(x_true[::3])))
    
    t0 = time.time()
    x_sol, info, obj_evals, grad_evals = solve_ipopt_fixed(
        n_sources, theta, data, K, verbose=verbose
    )
    elapsed = time.time() - t0
    
    obj = np.sum((forward_model(theta, x_sol, n_sources) - data)**2)
    pos_mean, pos_max = compute_errors(x_true, x_sol, n_sources)
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Objective: {obj:.2e}")
    print(f"  Position error: mean={pos_mean:.2e}, max={pos_max:.2e}")
    print(f"  Iterations: {info.get('iter_count', 'N/A')}")
    print(f"  IPOPT status: {info['status']} - {info['status_msg']}")
    
    return {'n': n_sources, 'obj': obj, 'pos': pos_mean, 'status': info['status']}


if __name__ == '__main__':
    import sys
    
    if not HAS_CYIPOPT:
        print("\ncyipopt not installed. Install with:")
        print("  conda install -c conda-forge cyipopt")
        print("\nAlternatively, use inverse_source_final.py which uses scipy.optimize")
        sys.exit(0)
    
    verbose = '-v' in sys.argv
    
    print("IPOPT FIXED - with interior point initialization")
    print("Key fix: push_to_interior() mimics MATLAB's behavior")
    
    results = []
    for n in [6, 8, 10, 12]:
        r = test(n, verbose=verbose)
        results.append(r)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'N':>4} | {'Objective':>12} | {'Pos Error':>12} | {'Status':>6}")
    print("-"*50)
    for r in results:
        print(f"{r['n']:4d} | {r['obj']:12.2e} | {r['pos']:12.2e} | {r['status']:6d}")
    
    print("\nMATLAB target: 12 sources → Position ~1e-5")
