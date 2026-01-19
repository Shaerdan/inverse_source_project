"""
MATLAB fmincon reproduction using cyipopt (IPOPT)
Same test setup: 6, 8, 10, 12 sources at r=0.8
"""

import numpy as np
import cyipopt
import time


def forward_polar(theta, params, n_sources):
    """Forward model: boundary potential"""
    phi = np.zeros_like(theta)
    for i in range(n_sources):
        S = params[3*i]
        r = params[3*i + 1]
        th = params[3*i + 2]
        arg = 1 + r**2 - 2*r*np.cos(theta - th)
        arg = np.maximum(arg, 1e-14)
        phi += (S / (2*np.pi)) * np.log(np.sqrt(arg))
    return 2 * phi


class InverseSourceProblem:
    """IPOPT problem definition"""
    
    def __init__(self, n_sources, theta, u_measured, reg_switch=0, lam_J=1e-4):
        self.n_sources = n_sources
        self.theta = theta
        self.u_measured = u_measured
        self.n_vars = 3 * n_sources
        self.reg_switch = reg_switch  # 0 = none (default), 1 = gradient-based (for close sources)
        self.lam_J = lam_J
    
    def objective(self, x):
        """Least squares objective with optional regularization"""
        u = forward_polar(self.theta, x, self.n_sources)
        misfit = np.sum((u - self.u_measured)**2)
        
        if self.reg_switch == 0:
            # No regularization (use for well-separated sources)
            return misfit
        elif self.reg_switch == 1:
            # Gradient-based regularizer (for resolution / close sources problem)
            # integral of (r*sin(θ-θs) / (1 + r² - 2r*cos(θ-θs)))²
            reg = 0.0
            theta_int = np.linspace(-np.pi, np.pi, 1000)
            for i in range(self.n_sources):
                r = x[3*i + 1]
                th = x[3*i + 2]
                denom = 1 + r**2 - 2*r*np.cos(theta_int - th)
                denom = np.maximum(denom, 1e-14)
                integrand = (r * np.sin(theta_int - th) / denom)**2
                reg += (1/(2*np.pi)) * np.trapz(integrand, theta_int)
            return misfit + self.lam_J * reg
        else:
            return misfit
    
    def gradient(self, x):
        """Analytical gradient (only for reg_switch=0, otherwise approximate)"""
        n = self.n_sources
        theta = self.theta
        
        # Forward
        phi = np.zeros_like(theta)
        for i in range(n):
            S = x[3*i]
            r = x[3*i + 1]
            th = x[3*i + 2]
            arg = 1 + r**2 - 2*r*np.cos(theta - th)
            arg = np.maximum(arg, 1e-14)
            phi += (S / (2*np.pi)) * np.log(np.sqrt(arg))
        phi = 2 * phi
        
        residual = phi - self.u_measured
        
        # Gradient of misfit term
        grad = np.zeros(3*n)
        for i in range(n):
            S = x[3*i]
            r = x[3*i + 1]
            th = x[3*i + 2]
            
            cos_diff = np.cos(theta - th)
            sin_diff = np.sin(theta - th)
            arg = 1 + r**2 - 2*r*cos_diff
            arg = np.maximum(arg, 1e-14)
            
            dphi_dS = (1/np.pi) * np.log(np.sqrt(arg))
            dphi_dr = (S / np.pi) * (r - cos_diff) / arg
            dphi_dth = -(S / np.pi) * r * sin_diff / arg
            
            grad[3*i] = 2 * np.sum(residual * dphi_dS)
            grad[3*i + 1] = 2 * np.sum(residual * dphi_dr)
            grad[3*i + 2] = 2 * np.sum(residual * dphi_dth)
        
        # Note: gradient of regularization term not implemented
        # For reg_switch=1, IPOPT will use finite differences
        
        return grad
    
    def constraints(self, x):
        """sum(S) = 0"""
        return np.array([sum(x[3*i] for i in range(self.n_sources))])
    
    def jacobian(self, x):
        """Jacobian of constraints"""
        jac = np.zeros(self.n_vars)
        for i in range(self.n_sources):
            jac[3*i] = 1.0
        return jac


def create_sources(n_sources, r=0.8, seed=None, intensity_mode='alternating'):
    """
    Create test sources.
    
    Args:
        n_sources: number of sources
        r: radius for all sources
        seed: random seed
        intensity_mode: 'alternating' for +-1, 'random' for randn
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Evenly spaced angles with small perturbation
    angles = np.linspace(0, 2*np.pi, n_sources, endpoint=False)
    angles += np.random.uniform(-0.1, 0.1, n_sources)
    
    # Intensities
    if intensity_mode == 'alternating':
        intensities = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n_sources)])
        intensities[-1] -= np.sum(intensities)  # Ensure sum = 0
    else:
        intensities = np.random.randn(n_sources)
        intensities = intensities - np.mean(intensities)
    
    params = []
    for i in range(n_sources):
        params.extend([intensities[i], r, angles[i]])
    
    return np.array(params)


def solve_inverse(u_measured, n_sources, theta, seed=123, reg_switch=0, lam_J=1e-4):
    """Single shot IPOPT - same as MATLAB fmincon
    
    Args:
        reg_switch: 0 = no regularization (default, for well-separated sources)
                    1 = gradient-based (for resolution/close sources problem)
    """
    np.random.seed(seed)
    
    K = 5.0
    
    # Bounds: [S, r, θ] for each source
    lb = []
    ub = []
    for _ in range(n_sources):
        lb.extend([-K, -1.0, -np.pi])
        ub.extend([K, 1.0, np.pi])
    lb = np.array(lb)
    ub = np.array(ub)
    
    # Constraint bounds (equality: sum(S) = 0)
    cl = np.array([0.0])
    cu = np.array([0.0])
    
    # Initial guess: linspace like MATLAB original
    x0 = np.linspace(-4, 4, 3*n_sources)
    
    # Create problem
    problem = InverseSourceProblem(n_sources, theta, u_measured, reg_switch=reg_switch, lam_J=lam_J)
    
    # IPOPT options (matching MATLAB tolerances)
    nlp = cyipopt.Problem(
        n=3*n_sources,
        m=1,  # 1 equality constraint
        problem_obj=problem,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu
    )
    
    nlp.add_option('tol', 1e-16)
    nlp.add_option('max_iter', 30000)
    nlp.add_option('print_level', 0)  # Quiet
    nlp.add_option('sb', 'yes')  # Suppress banner
    
    x_sol, info = nlp.solve(x0)
    
    return x_sol, info


def extract_sources(params, n_sources):
    sources = []
    for i in range(n_sources):
        S = params[3*i]
        r = params[3*i + 1]
        th = params[3*i + 2]
        x = r * np.cos(th)
        y = r * np.sin(th)
        sources.append((x, y, S))
    return sources


def compute_errors(true_params, rec_params, n_sources):
    true_src = extract_sources(true_params, n_sources)
    rec_src = extract_sources(rec_params, n_sources)
    
    used = set()
    pos_err = []
    int_err = []
    
    for (xt, yt, St) in true_src:
        best_d = np.inf
        best_j = -1
        for j, (xr, yr, Sr) in enumerate(rec_src):
            if j in used:
                continue
            d = np.sqrt((xt-xr)**2 + (yt-yr)**2)
            if d < best_d:
                best_d = d
                best_j = j
        used.add(best_j)
        pos_err.append(best_d)
        int_err.append(abs(St - rec_src[best_j][2]))
    
    return np.mean(pos_err), np.max(pos_err), np.mean(int_err), np.max(int_err)


def test_n_sources(n_sources, seed=42, intensity_mode='alternating', reg_switch=0):
    print(f"\n{'='*60}")
    print(f"Testing {n_sources} sources (r=0.8, {intensity_mode} intensities, reg={reg_switch})")
    print('='*60)
    
    n_boundary = 1000
    theta = np.linspace(-np.pi, np.pi, n_boundary)
    
    true_params = create_sources(n_sources, r=0.8, seed=seed, intensity_mode=intensity_mode)
    u_measured = forward_polar(theta, true_params, n_sources)
    
    print("\nTrue sources:")
    for i in range(n_sources):
        S = true_params[3*i]
        r = true_params[3*i + 1]
        th = true_params[3*i + 2]
        print(f"  {i+1}: r={r:.4f}, θ={th:+.4f}, S={S:+.4f}")
    print(f"  Sum of intensities: {sum(true_params[3*i] for i in range(n_sources)):.2e}")
    
    print("\nSolving inverse problem...")
    t0 = time.time()
    x_sol, info = solve_inverse(u_measured, n_sources, theta, seed=123, reg_switch=reg_switch)
    elapsed = time.time() - t0
    
    obj = np.sum((forward_polar(theta, x_sol, n_sources) - u_measured)**2)
    
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Final objective: {obj:.2e}")
    print(f"  IPOPT status: {info['status']} ({info['status_msg']})")
    
    print("\nRecovered sources:")
    for i in range(n_sources):
        S = x_sol[3*i]
        r = x_sol[3*i + 1]
        th = x_sol[3*i + 2]
        print(f"  {i+1}: r={r:.4f}, θ={th:+.4f}, S={S:+.4f}")
    print(f"  Sum of intensities: {sum(x_sol[3*i] for i in range(n_sources)):.2e}")
    
    mean_pos, max_pos, mean_int, max_int = compute_errors(true_params, x_sol, n_sources)
    print(f"\nErrors:")
    print(f"  Position: mean={mean_pos:.2e}, max={max_pos:.2e}")
    print(f"  Intensity: mean={mean_int:.2e}, max={max_int:.2e}")
    
    return {'n': n_sources, 'obj': obj, 'pos': mean_pos, 'int': mean_int, 'time': elapsed}


if __name__ == '__main__':
    import sys
    
    # Parse command line: python test_cyipopt.py [alternating|random] [reg_switch]
    intensity_mode = 'alternating'
    reg_switch = 0  # Default: no regularization (for well-separated sources)
    
    if len(sys.argv) > 1:
        intensity_mode = sys.argv[1]
    if len(sys.argv) > 2:
        reg_switch = int(sys.argv[2])
    
    print("Python cyipopt (IPOPT) test")
    print("Single shot (no restarts) - same as MATLAB fmincon")
    print(f"Intensity mode: {intensity_mode}")
    print(f"Regularization: {reg_switch} (0=none, 1=gradient-based for close sources)")
    print("Polar parameterization: [S, r, θ] per source")
    
    results = []
    for n in [6, 8, 10, 12]:
        r = test_n_sources(n, seed=42, intensity_mode=intensity_mode, reg_switch=reg_switch)
        results.append(r)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'N':>4} | {'Objective':>12} | {'Pos Error':>12} | {'Int Error':>12} | {'Time':>8}")
    print("-"*60)
    for r in results:
        print(f"{r['n']:4d} | {r['obj']:12.2e} | {r['pos']:12.2e} | {r['int']:12.2e} | {r['time']:7.1f}s")
